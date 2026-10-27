-- Projection of LLM usage observations into metric points, following the
-- OpenTelemetry GenAI metric conventions and the portable metric profiles
-- implemented by libdatadog's libdd-ai-usage crate.
--
-- An observation is a plain table describing one provider call or one
-- gateway request. A projector checks it, normalizes its usage and returns
-- the metric points: { name, instrument, unit, value, attributes }.
-- It returns nil and an error code when the observation is rejected.
--
-- This is a port of the crate's rules for the observation fields a gateway
-- fills. It does no I/O and keeps no state.

local math_huge = math.huge
local floor = math.floor
local sfmt = string.format

local _M = {}

local PROVIDER_ATTEMPT = "gen_ai.client.provider_attempt@0.1.0"
local TOKEN_BREAKDOWN = "trajectory.gen_ai.client.token_breakdown@0.1.0"
local PROVIDER_STREAMING = "gen_ai.client.provider_streaming@0.1.0"
local GATEWAY_REQUEST = "trajectory.gen_ai.gateway.request@0.1.0"

_M.PROVIDER_ATTEMPT = PROVIDER_ATTEMPT
_M.TOKEN_BREAKDOWN = TOKEN_BREAKDOWN
_M.GATEWAY_REQUEST = GATEWAY_REQUEST

-- The metrics each profile defines.
_M.PROFILE_METRICS = {
    [PROVIDER_ATTEMPT] = {
        "gen_ai.client.inference.duration",
        "gen_ai.client.operation.duration",
        "gen_ai.client.inference.operation.input_tokens",
        "gen_ai.client.inference.operation.output_tokens",
        "gen_ai.client.inference.usage.input_tokens",
        "gen_ai.client.inference.usage.output_tokens",
        "gen_ai.client.inference.time_to_first_chunk",
        "trajectory.gen_ai.client.inference.usage.server_tool.requests",
        "trajectory.gen_ai.client.inference.usage.cost",
        "trajectory.gen_ai.client.operation.cost",
    },
    [TOKEN_BREAKDOWN] = {
        "gen_ai.client.inference.usage.cache_read.input_tokens",
        "gen_ai.client.inference.usage.cache_write.input_tokens",
        "gen_ai.client.inference.usage.reasoning.output_tokens",
        "trajectory.gen_ai.client.inference.usage.uncached.input_tokens",
    },
    [PROVIDER_STREAMING] = {
        "gen_ai.client.inference.time_per_output_chunk",
    },
    [GATEWAY_REQUEST] = {
        "trajectory.gen_ai.gateway.request.duration",
        "trajectory.gen_ai.gateway.request.provider_operations",
        "trajectory.gen_ai.gateway.request.retries",
        "trajectory.gen_ai.gateway.request.fallbacks",
        "trajectory.gen_ai.gateway.request.estimated_cost",
        "trajectory.gen_ai.gateway.cache.operations",
    },
}

-- 2^53-1: the largest count or nano-USD amount held exactly.
local MAX_SAFE_INTEGER = 9007199254740991
local NANO_PER_USD = 1000000000
local USD_CONVERSION_LIMIT = 1e7

local OBSERVATION_POINTS = {
    harness = true,
    harness_transcript = true,
    agent_framework = true,
    gateway = true,
    network_proxy = true,
    provider = true,
    telemetry_import = true,
}

local USAGE_OPERATIONS = { chat = true, text_completion = true, embeddings = true }
local MODALITIES = { "text", "image", "audio" }
local MODALITY_SET = { text = true, image = true, audio = true }

local NANO_USD = "{nanoUSD}"
local TOKEN = "{token}"

-- Numbers

local function is_finite(value)
    return value == value and value ~= math_huge and value ~= -math_huge
end

-- How a number reads as a count: "count", "above" (above 2^53-1, or a
-- number no double holds) or "malformed" (negative, fractional, not finite).
local function read_count(value)
    if value == math_huge then
        return "above"
    end
    if not is_finite(value) or value < 0 or value ~= floor(value) then
        return "malformed"
    end
    if value > MAX_SAFE_INTEGER then
        return "above"
    end
    return "count", value
end
_M.read_count = read_count

-- The shortest decimal that reads back as the same double, never with an
-- exponent, as Rust's f64 Display writes it.
local function expand_exponent(text)
    local sign, mantissa, exponent = text:match("^(-?)([%d%.]+)e([-+]?%d+)$")
    if not mantissa then
        return text
    end
    exponent = tonumber(exponent)
    local whole, fraction = mantissa:match("^(%d*)%.?(%d*)$")
    local digits = whole .. fraction
    local point = #whole + exponent
    local out
    if point <= 0 then
        out = "0." .. string.rep("0", -point) .. digits
    elseif point >= #digits then
        out = digits .. string.rep("0", point - #digits)
    else
        out = digits:sub(1, point) .. "." .. digits:sub(point + 1)
    end
    out = out:gsub("^0+(%d)", "%1")
    if out:find("%.") then
        out = out:gsub("0+$", ""):gsub("%.$", "")
    end
    return sign .. out
end

local function shortest(value)
    if value == 0 then
        if 1 / value < 0 then
            return "-0"
        end
        return "0"
    end
    if value ~= value then
        return "NaN"
    end
    if value == math_huge then
        return "inf"
    end
    if value == -math_huge then
        return "-inf"
    end
    for precision = 1, 17 do
        local text = sfmt("%." .. precision .. "g", value)
        if tonumber(text) == value then
            return expand_exponent(text)
        end
    end
    return expand_exponent(sfmt("%.17g", value))
end
_M.shortest = shortest

-- The nano-USD value of a USD amount, or nil when it is unusable. It takes
-- the shortest decimal of the amount and rounds it half up to nine decimal
-- places on the digits, never in floating point.
local function nano_usd_from_usd(usd)
    if not (usd >= 0 and usd < USD_CONVERSION_LIMIT) then
        return nil
    end
    local text = shortest(usd):gsub("^-", "")
    local whole, fraction = text:match("^(%d+)%.?(%d*)$")
    if not whole then
        return nil
    end
    local nanos = 0
    for i = 1, 9 do
        nanos = nanos * 10 + (tonumber(fraction:sub(i, i)) or 0)
    end
    local round_up = (tonumber(fraction:sub(10, 10)) or 0) >= 5 and 1 or 0
    local total = tonumber(whole) * NANO_PER_USD + nanos + round_up
    if total > MAX_SAFE_INTEGER then
        return nil
    end
    return total
end
_M.nano_usd_from_usd = nano_usd_from_usd

local function nano_usd_from_number(value)
    local reading, count = read_count(value)
    if reading == "count" then
        return count
    end
    return nil
end

-- Grammars

local function is_identifier(value)
    return type(value) == "string" and value:find("^[a-z][a-z0-9_.%-]*$") ~= nil
end
_M.is_identifier = is_identifier

-- 1 to 32 letters, digits, '.', '_' or '-': a context band or service tier.
local function is_context_band(value)
    return type(value) == "string" and #value >= 1 and #value <= 32 and value:find("^[A-Za-z0-9._%-]+$") ~= nil
end
_M.is_service_tier = is_context_band

-- Field types

local STRING_FIELDS = {
    "operation_name",
    "provider_name",
    "request_model",
    "response_model",
    "error_type",
    "observation_point",
    "input_basis",
    "output_basis",
    "input_token_source",
    "output_token_source",
    "cost_source",
    "service_tier",
    "context_band",
    "provider_operation_coverage",
}
local NUMBER_FIELDS = {
    "duration_seconds",
    "time_to_first_chunk_seconds",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "cache_write_5m_input_tokens",
    "cache_write_1h_input_tokens",
    "reasoning_output_tokens",
    "cost_usd",
    "cost_nano_usd",
    "estimated_cost_usd",
    "estimated_cost_nano_usd",
    "provider_operations",
    "retries",
    "fallbacks",
}
local BOOLEAN_FIELDS = { "streaming", "batch" }
local MODALITY_FIELDS = { "input_tokens_by_modality", "output_tokens_by_modality" }

local function check_types(o)
    if o == nil then
        return "required_field_missing"
    end
    if type(o) ~= "table" or o[1] ~= nil then
        return "field_type_invalid"
    end
    for _, name in ipairs(STRING_FIELDS) do
        if o[name] ~= nil and type(o[name]) ~= "string" then
            return "field_type_invalid"
        end
    end
    for _, name in ipairs(NUMBER_FIELDS) do
        if o[name] ~= nil and type(o[name]) ~= "number" then
            return "field_type_invalid"
        end
    end
    for _, name in ipairs(BOOLEAN_FIELDS) do
        if o[name] ~= nil and type(o[name]) ~= "boolean" then
            return "field_type_invalid"
        end
    end
    for _, name in ipairs(MODALITY_FIELDS) do
        local parts = o[name]
        if parts ~= nil then
            if type(parts) ~= "table" then
                return "field_type_invalid"
            end
            for _, value in pairs(parts) do
                if type(value) ~= "number" then
                    return "field_type_invalid"
                end
            end
        end
    end
    if o.cache_outcomes ~= nil then
        if type(o.cache_outcomes) ~= "table" then
            return "field_type_invalid"
        end
        for _, value in ipairs(o.cache_outcomes) do
            if type(value) ~= "string" then
                return "field_type_invalid"
            end
        end
    end
    return nil
end

-- Attributes

local function copy(t)
    local out = {}
    for k, v in pairs(t) do
        out[k] = v
    end
    return out
end

local function with(base, key, value)
    local out = copy(base)
    out[key] = value
    return out
end

local function with_some(base, key, value)
    if value == nil then
        return copy(base)
    end
    return with(base, key, value)
end

local function with_price_tags(base, tags)
    local out =
        with_some(with_some(base, "trajectory.context.band", tags.band), "trajectory.request.service_tier", tags.tier)
    if tags.batch then
        out["trajectory.request.batch"] = "true"
    end
    return out
end

local function with_token_source(base, estimated)
    if estimated then
        return with(base, "trajectory.token.source", "estimated")
    end
    return copy(base)
end

local function histogram(name, unit, value, attributes)
    return { name = name, instrument = "histogram", unit = unit, value = value, attributes = attributes }
end

local function counter(name, unit, value, attributes)
    return { name = name, instrument = "counter", unit = unit, value = value, attributes = attributes }
end

-- Usage

local COUNT_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "cache_write_5m_input_tokens",
    "cache_write_1h_input_tokens",
    "reasoning_output_tokens",
}

local function check_counts(o)
    for _, name in ipairs(COUNT_FIELDS) do
        if o[name] ~= nil and read_count(o[name]) == "malformed" then
            return "count_invalid"
        end
    end
    for _, name in ipairs(MODALITY_FIELDS) do
        local parts = o[name]
        if parts then
            for _, modality in ipairs(MODALITIES) do
                if parts[modality] ~= nil and read_count(parts[modality]) == "malformed" then
                    return "count_invalid"
                end
            end
        end
    end
    return nil
end

local function check_values(o)
    if o.input_basis ~= nil and o.input_basis ~= "includes_cache" and o.input_basis ~= "excludes_cache" then
        return "value_not_allowed"
    end
    if o.output_basis ~= nil and o.output_basis ~= "includes_reasoning" and o.output_basis ~= "excludes_reasoning" then
        return "value_not_allowed"
    end
    for _, name in ipairs(MODALITY_FIELDS) do
        local parts = o[name]
        if parts then
            for key in pairs(parts) do
                if not MODALITY_SET[key] then
                    return "value_not_allowed"
                end
            end
        end
    end
    return nil
end

local function bounded(value)
    if value == nil then
        return nil
    end
    local reading, count = read_count(value)
    if reading == "count" then
        return count
    end
    return nil
end

local function bounded_sum(parts)
    local total = 0
    for _, part in ipairs(parts) do
        total = total + part
    end
    if total > MAX_SAFE_INTEGER then
        return nil
    end
    return total
end

local function bounded_parts(parts)
    local out = {}
    for _, modality in ipairs(MODALITIES) do
        local count = parts and bounded(parts[modality])
        if count ~= nil then
            out[#out + 1] = { modality, count }
        end
    end
    return out
end

local function sum_parts(parts)
    local total = 0
    for _, part in ipairs(parts) do
        total = total + part[2]
    end
    return total
end

-- The Counter increments of one total: its reported parts and the remainder
-- as unknown, or the parts alone when the total is not known.
local function by_modality(total, parts)
    if total == nil then
        return parts
    end
    if #parts == 0 then
        return { { "unknown", total } }
    end
    local remainder = total - sum_parts(parts)
    local increments = parts
    if remainder > 0 then
        increments[#increments + 1] = { "unknown", remainder }
    end
    return increments
end

-- Normalize the usage of one operation: inclusive totals (input includes
-- cached input, output includes reasoning), cache parts and modality
-- increments. Inconsistent sides are left out.
local function normalize_usage(o, embeddings)
    local input = bounded(o.input_tokens)
    local output = bounded(o.output_tokens)
    local read = bounded(o.cache_read_input_tokens)
    local write = bounded(o.cache_write_input_tokens)
    local write_5m = bounded(o.cache_write_5m_input_tokens)
    local write_1h = bounded(o.cache_write_1h_input_tokens)
    local reasoning = bounded(o.reasoning_output_tokens)
    local input_parts = bounded_parts(o.input_tokens_by_modality)
    local output_parts = bounded_parts(o.output_tokens_by_modality)
    local input_basis = o.input_basis
    local output_basis = o.output_basis

    local cache_part_known = read ~= nil or write ~= nil or write_5m ~= nil or write_1h ~= nil
    local function in_sums(part)
        if embeddings and part == nil then
            return 0
        end
        return part
    end

    local inclusive_input
    if not cache_part_known or input_basis == "includes_cache" then
        inclusive_input = input
    elseif input_basis == "excludes_cache" then
        local r, w = in_sums(read), in_sums(write)
        if input ~= nil and r ~= nil and w ~= nil then
            inclusive_input = bounded_sum({ input, r, w })
        end
    end

    local inclusive_output
    if reasoning == nil or output_basis == "includes_reasoning" then
        inclusive_output = output
    elseif output_basis == "excludes_reasoning" and output ~= nil then
        inclusive_output = bounded_sum({ output, reasoning })
    end

    local lifetime_parts = (write_5m or 0) + (write_1h or 0)
    local cache_exceeds_input = false
    if cache_part_known and input_basis == "includes_cache" and input ~= nil then
        local writes = write or lifetime_parts
        cache_exceeds_input = (read or 0) + writes > input
    end
    local lifetimes_exceed_writes = write ~= nil and lifetime_parts > write
    local function exceeds(parts, total)
        return total ~= nil and sum_parts(parts) > total
    end
    local input_inconsistent = cache_exceeds_input or lifetimes_exceed_writes or exceeds(input_parts, inclusive_input)
    local output_inconsistent = (
        output_basis == "includes_reasoning"
        and reasoning ~= nil
        and output ~= nil
        and reasoning > output
    ) or exceeds(output_parts, inclusive_output)

    local usage = { input_by_modality = {}, output_by_modality = {} }
    if not input_inconsistent then
        usage.input_total = inclusive_input
        usage.cache_read = read
        usage.cache_write_5m = write_5m
        usage.cache_write_1h = write_1h
        if write ~= nil then
            local remainder = write - lifetime_parts
            if remainder >= 0 and (remainder > 0 or (write_5m == nil and write_1h == nil)) then
                usage.cache_write_unspecified = remainder
            end
        end
        if not cache_part_known then
            if embeddings then
                usage.uncached_input = input
            end
        elseif input_basis == "excludes_cache" then
            usage.uncached_input = input
        elseif input_basis == "includes_cache" then
            local r, w = in_sums(read), in_sums(write)
            if inclusive_input ~= nil and r ~= nil and w ~= nil and inclusive_input - r - w >= 0 then
                usage.uncached_input = inclusive_input - r - w
            end
        end
        usage.input_by_modality = by_modality(inclusive_input, input_parts)
    end
    if not output_inconsistent then
        usage.output_total = inclusive_output
        usage.reasoning = reasoning
        usage.output_by_modality = by_modality(inclusive_output, output_parts)
    end
    return usage
end

-- Checks

local function check_operation(name)
    if name == nil then
        return nil, "required_field_missing"
    end
    if not USAGE_OPERATIONS[name] then
        return nil, "value_not_allowed"
    end
    return name
end

local function check_provider(name)
    if name == nil then
        return nil, "required_field_missing"
    end
    if not is_identifier(name) then
        return nil, "identifier_invalid"
    end
    return name
end

local function is_duration(value)
    return type(value) == "number" and is_finite(value) and value >= 0
end

local function check_token_sources(o)
    for _, name in ipairs({ "input_token_source", "output_token_source" }) do
        local value = o[name]
        if value ~= nil and value ~= "reported" and value ~= "estimated" then
            return "value_not_allowed"
        end
    end
    return nil
end

local function check_price_tags(o)
    if o.context_band ~= nil and not is_context_band(o.context_band) then
        return nil, "identifier_invalid"
    end
    if o.service_tier ~= nil and not is_context_band(o.service_tier) then
        return nil, "identifier_invalid"
    end
    return { band = o.context_band, tier = o.service_tier, batch = o.batch == true }
end

local function check_observation_point(point)
    if point == nil then
        return "required_field_missing"
    end
    if not OBSERVATION_POINTS[point] then
        return "value_not_allowed"
    end
    return nil
end

local function check_usage(o)
    return check_counts(o) or check_values(o) or check_token_sources(o)
end

-- The cost of an observation after the contradiction checks. Returns
-- false when the observation states no cost, and the amount as nil when
-- it is unusable.
local function check_cost(o)
    local has_cost = o.cost_usd ~= nil or o.cost_nano_usd ~= nil
    local source = o.cost_source
    if has_cost and source == nil then
        return nil, "fields_contradict"
    end
    if source ~= nil and not has_cost then
        return nil, "fields_contradict"
    end
    if source ~= nil and source ~= "reported" and source ~= "calculated" and source ~= "estimated" then
        return nil, "value_not_allowed"
    end
    if source == nil then
        return false
    end
    local from_nano = o.cost_nano_usd ~= nil and (nano_usd_from_number(o.cost_nano_usd) or false) or nil
    local from_usd = o.cost_usd ~= nil and (nano_usd_from_usd(o.cost_usd) or false) or nil
    if from_nano and from_usd and from_nano ~= from_usd then
        return nil, "fields_contradict"
    end
    if source == "calculated" and (o.input_token_source == "estimated" or o.output_token_source == "estimated") then
        return nil, "fields_contradict"
    end
    local cost = { source = source }
    if from_nano == false or from_usd == false then
        return cost
    end
    if from_nano then
        cost.nano = from_nano
        cost.usd = o.cost_usd or from_nano / NANO_PER_USD
    elseif from_usd then
        cost.nano = from_usd
        cost.usd = o.cost_usd
    end
    return cost
end

local function provider_attributes(operation, provider, o)
    return {
        ["gen_ai.operation.name"] = operation,
        ["gen_ai.provider.name"] = provider,
        ["gen_ai.request.model"] = o.request_model,
        ["gen_ai.response.model"] = o.response_model,
        ["trajectory.observation.point"] = o.observation_point,
    }
end

local function token_points(points, direction, total, increments, base, tags, estimated)
    local sourced = with_token_source(base, estimated)
    if total ~= nil then
        points[#points + 1] =
            histogram("gen_ai.client.inference.operation." .. direction .. "_tokens", TOKEN, total, copy(sourced))
    end
    local counter_base = with_price_tags(sourced, tags)
    for _, increment in ipairs(increments) do
        points[#points + 1] = counter(
            "gen_ai.client.inference.usage." .. direction .. "_tokens",
            TOKEN,
            increment[2],
            with(counter_base, "gen_ai.token.modality", increment[1])
        )
    end
end

-- Project one provider operation (gen_ai.client.provider_attempt@0.1.0).
function _M.project_provider_attempt(o)
    local err = check_types(o) or check_usage(o)
    if err then
        return nil, err
    end
    local operation, provider
    operation, err = check_operation(o.operation_name)
    if err then
        return nil, err
    end
    provider, err = check_provider(o.provider_name)
    if err then
        return nil, err
    end
    if not is_duration(o.duration_seconds) then
        return nil, "duration_invalid"
    end
    if o.streaming == nil then
        return nil, "required_field_missing"
    end
    if o.time_to_first_chunk_seconds ~= nil then
        if not is_duration(o.time_to_first_chunk_seconds) then
            return nil, "duration_invalid"
        end
        if not o.streaming then
            return nil, "fields_contradict"
        end
    end
    local tags
    tags, err = check_price_tags(o)
    if err then
        return nil, err
    end
    local cost
    cost, err = check_cost(o)
    if err then
        return nil, err
    end
    if o.error_type ~= nil and not is_identifier(o.error_type) then
        return nil, "identifier_invalid"
    end
    err = check_observation_point(o.observation_point)
    if err then
        return nil, err
    end

    local usage = normalize_usage(o, operation == "embeddings")
    local base = provider_attributes(operation, provider, o)
    local duration_metric = operation == "embeddings" and "gen_ai.client.operation.duration"
        or "gen_ai.client.inference.duration"
    local points = {
        histogram(duration_metric, "s", o.duration_seconds, with_some(base, "error.type", o.error_type)),
    }
    local input_estimated = o.input_token_source == "estimated"
    local output_estimated = o.output_token_source == "estimated"
    token_points(points, "input", usage.input_total, usage.input_by_modality, base, tags, input_estimated)
    token_points(points, "output", usage.output_total, usage.output_by_modality, base, tags, output_estimated)
    if o.time_to_first_chunk_seconds ~= nil then
        points[#points + 1] =
            histogram("gen_ai.client.inference.time_to_first_chunk", "s", o.time_to_first_chunk_seconds, copy(base))
    end
    if cost and cost.nano then
        local sourced = with(base, "trajectory.cost.source", cost.source)
        points[#points + 1] = counter(
            "trajectory.gen_ai.client.inference.usage.cost",
            NANO_USD,
            cost.nano,
            with_price_tags(sourced, tags)
        )
        points[#points + 1] = histogram("trajectory.gen_ai.client.operation.cost", "{USD}", cost.usd, sourced)
    end
    return points
end

-- Project the cache and reasoning parts of one provider operation
-- (trajectory.gen_ai.client.token_breakdown@0.1.0).
function _M.project_token_breakdown(o)
    local err = check_types(o) or check_usage(o)
    if err then
        return nil, err
    end
    local operation, provider
    operation, err = check_operation(o.operation_name)
    if err then
        return nil, err
    end
    provider, err = check_provider(o.provider_name)
    if err then
        return nil, err
    end
    local embeddings = operation == "embeddings"
    if o.input_tokens == nil then
        return nil, "required_field_missing"
    end
    if o.output_tokens == nil and not embeddings then
        return nil, "required_field_missing"
    end
    local tags
    tags, err = check_price_tags(o)
    if err then
        return nil, err
    end
    err = check_observation_point(o.observation_point)
    if err then
        return nil, err
    end

    local usage = normalize_usage(o, embeddings)
    local base =
        with_price_tags(with(provider_attributes(operation, provider, o), "gen_ai.token.modality", "unknown"), tags)
    local input_side = with_token_source(base, o.input_token_source == "estimated")
    local points = {}
    if usage.cache_read ~= nil then
        points[#points + 1] =
            counter("gen_ai.client.inference.usage.cache_read.input_tokens", TOKEN, usage.cache_read, copy(input_side))
    end
    for _, lifetime in ipairs({
        { "5m", usage.cache_write_5m },
        { "1h", usage.cache_write_1h },
        { "unspecified", usage.cache_write_unspecified },
    }) do
        if lifetime[2] ~= nil then
            points[#points + 1] = counter(
                "gen_ai.client.inference.usage.cache_write.input_tokens",
                TOKEN,
                lifetime[2],
                with(input_side, "trajectory.cache.lifetime", lifetime[1])
            )
        end
    end
    if usage.uncached_input ~= nil then
        points[#points + 1] = counter(
            "trajectory.gen_ai.client.inference.usage.uncached.input_tokens",
            TOKEN,
            usage.uncached_input,
            copy(input_side)
        )
    end
    if usage.reasoning ~= nil then
        points[#points + 1] = counter(
            "gen_ai.client.inference.usage.reasoning.output_tokens",
            TOKEN,
            usage.reasoning,
            with_token_source(base, o.output_token_source == "estimated")
        )
    end
    return points
end

local function check_plain_count(value)
    if value == nil then
        return nil
    end
    if is_finite(value) and read_count(value) ~= "malformed" then
        return value
    end
    return nil, "count_invalid"
end

-- Project one client request to the gateway
-- (trajectory.gen_ai.gateway.request@0.1.0).
function _M.project_gateway_request(o)
    local err = check_types(o)
    if err then
        return nil, err
    end
    local operation
    operation, err = check_operation(o.operation_name)
    if err then
        return nil, err
    end
    if not is_duration(o.duration_seconds) then
        return nil, "duration_invalid"
    end
    local operations, retries, fallbacks
    operations, err = check_plain_count(o.provider_operations)
    if err then
        return nil, err
    end
    retries, err = check_plain_count(o.retries)
    if err then
        return nil, err
    end
    fallbacks, err = check_plain_count(o.fallbacks)
    if err then
        return nil, err
    end
    local coverage = o.provider_operation_coverage
    if (operations ~= nil) ~= (coverage ~= nil) then
        return nil, "fields_contradict"
    end
    if coverage ~= nil and coverage ~= "complete" and coverage ~= "partial" then
        return nil, "value_not_allowed"
    end
    local outcomes = o.cache_outcomes or {}
    for _, outcome in ipairs(outcomes) do
        if outcome ~= "hit" and outcome ~= "miss" and outcome ~= "write" then
            return nil, "value_not_allowed"
        end
    end
    local from_nano = o.estimated_cost_nano_usd ~= nil and (nano_usd_from_number(o.estimated_cost_nano_usd) or false)
        or nil
    local from_usd = o.estimated_cost_usd ~= nil and (nano_usd_from_usd(o.estimated_cost_usd) or false) or nil
    if from_nano and from_usd and from_nano ~= from_usd then
        return nil, "fields_contradict"
    end
    if o.error_type ~= nil and not is_identifier(o.error_type) then
        return nil, "identifier_invalid"
    end
    if o.observation_point ~= nil and o.observation_point ~= "gateway" then
        return nil, "value_not_allowed"
    end

    local base = {
        ["gen_ai.operation.name"] = operation,
        ["trajectory.observation.point"] = "gateway",
        ["gen_ai.request.model"] = o.request_model,
    }
    local points = {
        histogram(
            "trajectory.gen_ai.gateway.request.duration",
            "s",
            o.duration_seconds,
            with_some(base, "error.type", o.error_type)
        ),
    }
    if operations ~= nil then
        points[#points + 1] = histogram(
            "trajectory.gen_ai.gateway.request.provider_operations",
            "{operation}",
            operations,
            with(base, "trajectory.provider.operation.coverage", coverage)
        )
    end
    if retries ~= nil then
        points[#points + 1] = histogram("trajectory.gen_ai.gateway.request.retries", "{retry}", retries, copy(base))
    end
    if fallbacks ~= nil then
        points[#points + 1] =
            histogram("trajectory.gen_ai.gateway.request.fallbacks", "{fallback}", fallbacks, copy(base))
    end
    local cache_counts, cache_order = {}, {}
    for _, outcome in ipairs(outcomes) do
        if not cache_counts[outcome] then
            cache_order[#cache_order + 1] = outcome
            cache_counts[outcome] = 0
        end
        cache_counts[outcome] = cache_counts[outcome] + 1
    end
    for _, outcome in ipairs(cache_order) do
        points[#points + 1] = counter(
            "trajectory.gen_ai.gateway.cache.operations",
            "{operation}",
            cache_counts[outcome],
            with(base, "trajectory.cache.outcome", outcome)
        )
    end
    if from_nano ~= false and from_usd ~= false and (from_nano or from_usd) then
        points[#points + 1] =
            counter("trajectory.gen_ai.gateway.request.estimated_cost", NANO_USD, from_nano or from_usd, copy(base))
    end
    return points
end

-- Deployment attributes: the opt-in dimensions a deployment adds after
-- projection. A point-carried attribute goes on every point that does not
-- already carry it; a resource-carried one is returned apart.
local DEPLOYMENT_ATTRIBUTES = {
    ["host.name"] = "resource",
    ["service.name"] = "resource",
    ["trajectory.client_source"] = "point",
    ["trajectory.gateway.destination.id"] = "point",
    ["trajectory.gateway.key.alias"] = "point",
    ["trajectory.gateway.route"] = "point",
    ["trajectory.request.api_key_id"] = "point",
    ["trajectory.request.project_id"] = "point",
    ["trajectory.request.provider_account_id"] = "point",
    ["trajectory.team.id"] = "point",
    ["user.id"] = "point",
}

-- An identifier, never free text: ^[A-Za-z0-9][A-Za-z0-9._@+:/'|-]{0,127}$
local function is_deployment_value(value)
    return type(value) == "string"
        and #value >= 1
        and #value <= 128
        and value:find("^[A-Za-z0-9][A-Za-z0-9._@+:/'|%-]*$") ~= nil
end
_M.is_deployment_value = is_deployment_value

-- Returns the points, with the point-carried attributes added, and the
-- resource attributes; or nil and an error code for an unknown name.
function _M.with_deployment_attributes(points, attributes)
    local resource = {}
    if not attributes then
        return points, resource
    end
    local names = {}
    for name in pairs(attributes) do
        if not DEPLOYMENT_ATTRIBUTES[name] then
            return nil, "deployment_attribute_unknown"
        end
        names[#names + 1] = name
    end
    table.sort(names)
    for _, name in ipairs(names) do
        local value = attributes[name]
        if is_deployment_value(value) then
            if DEPLOYMENT_ATTRIBUTES[name] == "resource" then
                resource[name] = value
            else
                for _, point in ipairs(points) do
                    if point.attributes[name] == nil then
                        point.attributes[name] = value
                    end
                end
            end
        end
    end
    return points, resource
end

return _M
