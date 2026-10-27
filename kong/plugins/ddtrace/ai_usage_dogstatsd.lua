-- The Datadog binding of the LLM usage metric profiles: the series submitted
-- for a profile's points, and their DogStatsD encoding. The output is the
-- same as libdatadog's libdd-ai-usage DogStatsD binding, so series are the
-- same whichever gateway sends them.
--
-- - A metric keeps its name. A Counter becomes a Count; a Histogram becomes
--   one Distribution sample per input sample, plus {name}.sum and
--   {name}.count Counts.
-- - A Counter in {nanoUSD} is summed exactly and submitted in USD.
-- - Every series carries trajectory.profile:{id}/{version}.
-- - A tag key is the attribute name, except service.name and host.name,
--   which become service and host. Tag values are normalized as Datadog
--   stores them.

local projection = require("kong.plugins.ddtrace.ai_usage_projection")

local shortest = projection.shortest
local PROFILE_METRICS = projection.PROFILE_METRICS
local byte = string.byte
local schar = string.char
local concat = table.concat
local sort = table.sort
local floor = math.floor

local PROFILE_ATTRIBUTE = "trajectory.profile"
local NANO_USD_PER_USD = 1e9

local _M = {}

local function tag_key(attribute)
    if attribute == "service.name" then
        return "service"
    elseif attribute == "host.name" then
        return "host"
    end
    return attribute
end

local function is_renamed(attribute)
    return tag_key(attribute) ~= attribute
end

-- The length of the UTF-8 code point starting at byte i, or 1 for a byte
-- that does not start a valid one.
local function code_point_length(value, i, b)
    local length
    if b < 0xC2 then
        return 1
    elseif b < 0xE0 then
        length = 2
    elseif b < 0xF0 then
        length = 3
    elseif b < 0xF5 then
        length = 4
    else
        return 1
    end
    for j = i + 1, i + length - 1 do
        local c = byte(value, j)
        if not c or c < 0x80 or c > 0xBF then
            return 1
        end
    end
    return length
end

-- Normalize a tag value as Datadog stores it: ASCII upper case becomes lower
-- case, and every code point outside a-z, 0-9, _, -, :, . and / becomes one
-- _. Nothing is truncated, collapsed or trimmed.
local function datadog_tag_value(value)
    local out = {}
    local i, n = 1, #value
    while i <= n do
        local b = byte(value, i)
        if b >= 0x41 and b <= 0x5A then
            out[#out + 1] = schar(b + 32)
            i = i + 1
        elseif
            (b >= 0x61 and b <= 0x7A)
            or (b >= 0x30 and b <= 0x39)
            or b == 0x5F -- _
            or b == 0x2D -- -
            or b == 0x3A -- :
            or b == 0x2E -- .
            or b == 0x2F -- /
        then
            out[#out + 1] = schar(b)
            i = i + 1
        else
            out[#out + 1] = "_"
            i = i + code_point_length(value, i, b)
        end
    end
    return concat(out)
end
_M.datadog_tag_value = datadog_tag_value

local function is_versioned(attribute)
    return attribute == PROFILE_ATTRIBUTE or attribute == "trajectory.cost.rate_card"
end

local function tag_value(attribute, value)
    if is_versioned(attribute) then
        value = value:gsub("@", "/")
    end
    return datadog_tag_value(value)
end

-- The tags of one carrier, keyed by tag key. When two attributes give one
-- key, the attribute this binding renames wins.
local function carrier_tags(attributes, tags)
    for attribute, value in pairs(attributes) do
        local key = tag_key(attribute)
        local existing = tags[key]
        if not (existing and existing.carrier == attributes and is_renamed(existing.attribute)) then
            tags[key] = { attribute = attribute, value = value, carrier = attributes }
        end
    end
end

local function series_tags(profile, point, resource)
    local stated = point.attributes[PROFILE_ATTRIBUTE]
    if stated ~= nil and stated ~= profile then
        return nil, "metric_profile_mismatch"
    end
    local tags = {}
    carrier_tags(resource or {}, tags)
    tags[PROFILE_ATTRIBUTE] = nil
    local from_point = {}
    carrier_tags(point.attributes, from_point)
    for key, tag in pairs(from_point) do
        tags[key] = tag
    end
    tags[PROFILE_ATTRIBUTE] = { attribute = PROFILE_ATTRIBUTE, value = profile }
    local keys = {}
    for key in pairs(tags) do
        keys[#keys + 1] = key
    end
    sort(keys)
    local out = {}
    for i, key in ipairs(keys) do
        local tag = tags[key]
        out[i] = key .. ":" .. tag_value(tag.attribute, tag.value)
    end
    return out
end

local profile_sets = {}
for profile, names in pairs(PROFILE_METRICS) do
    local set = {}
    for _, name in ipairs(names) do
        set[name] = true
    end
    profile_sets[profile] = set
end

local batch_methods = {}
local batch_mt = { __index = batch_methods }

-- A batch collects the series of one submission. Counts are summed by name
-- and tag set; each Histogram sample is kept.
function _M.new_batch()
    return setmetatable({ counts = {}, count_list = {}, distributions = {}, size = 0 }, batch_mt)
end

-- Whole-number increments are added exactly, as 64-bit integers; a sum that
-- gets a fractional increment continues as a double.
local function is_whole(value)
    return value == value and value >= 0 and value < 2 ^ 63 and value == floor(value)
end

function batch_methods:count(name, tags, increment, scaled)
    local key = name .. "\n" .. concat(tags, "\n")
    local entry = self.counts[key]
    if not entry then
        entry = { name = name, tags = tags, total = 0LL, exact = true, scaled = scaled }
        self.counts[key] = entry
        self.count_list[#self.count_list + 1] = entry
    end
    if entry.exact and is_whole(increment) then
        entry.total = entry.total + increment
    else
        if entry.exact then
            entry.exact = false
            entry.total = tonumber(entry.total)
        end
        entry.total = entry.total + increment
    end
end

-- Add the points of one profile. Every point must be a metric of the
-- profile; on a mismatch nothing is added.
function batch_methods:add(profile, points, resource, historical)
    local names = profile_sets[profile]
    if not names then
        return nil, "profile_invalid"
    end
    local tagged = {}
    for i, point in ipairs(points) do
        if not names[point.name] then
            return nil, "metric_profile_mismatch"
        end
        local tags, err = series_tags(profile, point, resource)
        if not tags then
            return nil, err
        end
        tagged[i] = tags
    end
    for i, point in ipairs(points) do
        local tags = tagged[i]
        if point.instrument == "counter" then
            self:count(point.name, tags, point.value, point.unit == "{nanoUSD}")
        else
            if not historical then
                self.distributions[#self.distributions + 1] =
                    { name = point.name, type = "distribution", value = point.value, tags = tags }
            end
            self:count(point.name .. ".sum", tags, point.value, false)
            self:count(point.name .. ".count", tags, 1, false)
        end
        self.size = self.size + 1
    end
    return true
end

function batch_methods:is_empty()
    return self.size == 0
end

local function compare_tags(a, b)
    local n = math.min(#a, #b)
    for i = 1, n do
        if a[i] ~= b[i] then
            return a[i] < b[i]
        end
    end
    return #a < #b
end

local function series_less(a, b)
    if a.name ~= b.name then
        return a.name < b.name
    end
    local at, bt = concat(a.tags, "\n"), concat(b.tags, "\n")
    if at ~= bt then
        return compare_tags(a.tags, b.tags)
    end
    return a.value < b.value
end

-- The series of the batch, sorted by name, then tags, then value.
function batch_methods:series()
    local out = {}
    for _, s in ipairs(self.distributions) do
        out[#out + 1] = s
    end
    for _, entry in ipairs(self.count_list) do
        local value = tonumber(entry.total)
        if entry.scaled then
            value = value / NANO_USD_PER_USD
        end
        out[#out + 1] = { name = entry.name, type = "count", value = value, tags = entry.tags }
    end
    sort(out, series_less)
    return out
end

-- The DogStatsD line of one series point, without a trailing newline.
local function dogstatsd_line(s)
    local line = s.name .. ":" .. shortest(s.value) .. "|" .. (s.type == "count" and "c" or "d")
    if #s.tags > 0 then
        line = line .. "|#" .. concat(s.tags, ",")
    end
    return line
end
_M.dogstatsd_line = dogstatsd_line

function batch_methods:lines()
    local out = {}
    for i, s in ipairs(self:series()) do
        out[i] = dogstatsd_line(s)
    end
    return out
end

return _M
