local cjson = require("cjson.safe")
local projection = require("kong.plugins.ddtrace.ai_usage_projection")
local dogstatsd = require("kong.plugins.ddtrace.ai_usage_dogstatsd")

local function fixture_path()
    local source = debug.getinfo(1, "S").source:sub(2)
    return source:gsub("01%-unit%-tests/[^/]+$", "fixtures/ai_usage_conformance.json")
end

-- JSON null is an absent member.
local function strip_nulls(value)
    if type(value) ~= "table" then
        return value
    end
    local out = {}
    for k, v in pairs(value) do
        if v ~= cjson.null then
            out[k] = strip_nulls(v)
        end
    end
    return out
end

local function load_fixtures()
    local f = assert(io.open(fixture_path(), "r"))
    local text = f:read("*a")
    f:close()
    return strip_nulls(assert(cjson.decode(text)))
end

local PROJECTORS = {
    [projection.PROVIDER_ATTEMPT] = projection.project_provider_attempt,
    [projection.TOKEN_BREAKDOWN] = projection.project_token_breakdown,
    [projection.GATEWAY_REQUEST] = projection.project_gateway_request,
}

local function attribute_key(attributes)
    local keys = {}
    for k in pairs(attributes or {}) do
        keys[#keys + 1] = k
    end
    table.sort(keys)
    local parts = {}
    for i, k in ipairs(keys) do
        parts[i] = k .. "=" .. attributes[k]
    end
    return table.concat(parts, ",")
end

-- Counters are aggregated per name and attribute set; Histograms are
-- samples. Returns sorted strings that compare as multisets.
local function canonical(points)
    local counters, order, out = {}, {}, {}
    for _, p in ipairs(points) do
        local id = p.name .. "|" .. p.instrument .. "|" .. p.unit .. "|" .. attribute_key(p.attributes)
        if p.instrument == "counter" then
            if not counters[id] then
                order[#order + 1] = id
                counters[id] = 0
            end
            counters[id] = counters[id] + p.value
        else
            -- Values compare as numbers: -0 is 0.
            out[#out + 1] = id .. "|" .. projection.shortest(p.value + 0)
        end
    end
    for _, id in ipairs(order) do
        out[#out + 1] = id .. "|" .. projection.shortest(counters[id])
    end
    table.sort(out)
    return out
end

local function project(case)
    local projector = PROJECTORS[case.profile]
    if not projector then
        return nil, "profile_invalid"
    end
    local points, err = projector(case.observation)
    if not points then
        return nil, err
    end
    return projection.with_deployment_attributes(points, case.deployment_attributes)
end

describe("ai usage projection conformance", function()
    local fixtures = load_fixtures()

    it("runs most of the conformance cases", function()
        assert.is_true(#fixtures.projection >= 200)
    end)

    for _, case in ipairs(fixtures.projection) do
        it(case.profile .. " " .. case.case, function()
            local points, resource = project(case)
            if case.expected_error_code then
                assert.is_nil(points)
                assert.equal(case.expected_error_code, resource)
                return
            end
            assert.is_table(points, tostring(resource))
            assert.same(canonical(case.expected_metrics or {}), canonical(points))
            assert.same(case.expected_resource_attributes or {}, resource)
        end)
    end
end)

describe("ai usage datadog binding conformance", function()
    local fixtures = load_fixtures()

    for _, case in ipairs(fixtures.datadog) do
        it(case.case, function()
            local batch = dogstatsd.new_batch()
            local ok, err = batch:add(case.profile, case.points, case.resource_attributes, case.historical)
            if case.expected_error_code then
                assert.is_nil(ok)
                assert.equal(case.expected_error_code, err)
                assert.is_true(batch:is_empty())
                return
            end
            assert.is_true(ok, err)
            local series = batch:series()
            assert.equal(#case.expected_series, #series)
            for i, expected in ipairs(case.expected_series) do
                local got = series[i]
                assert.equal(expected.name, got.name)
                assert.equal(expected.type, got.type)
                assert.equal(expected.value, got.value)
                assert.same(expected.tags, got.tags)
            end
        end)
    end
end)

describe("ai usage numbers", function()
    it("writes the shortest decimal without an exponent", function()
        assert.equal("12", projection.shortest(12))
        assert.equal("0.3", projection.shortest(0.3))
        assert.equal("0.0000000001", projection.shortest(1e-10))
        assert.equal("0.30000000000000004", projection.shortest(0.1 + 0.2))
        assert.equal("0.0000057", projection.shortest(5.7e-6))
        assert.equal("100000000000000000000", projection.shortest(1e20))
        assert.equal("1.5", projection.shortest(1.5))
        assert.equal("-2.25", projection.shortest(-2.25))
        assert.equal("0", projection.shortest(0))
    end)

    it("converts USD to nano-USD on the decimal digits", function()
        assert.equal(1005000001, projection.nano_usd_from_usd(1.0050000005))
        assert.equal(1000001, projection.nano_usd_from_usd(0.0010000005))
        assert.equal(0, projection.nano_usd_from_usd(4e-10))
        assert.equal(1, projection.nano_usd_from_usd(5e-10))
        assert.is_nil(projection.nano_usd_from_usd(-1))
        assert.is_nil(projection.nano_usd_from_usd(1e7))
    end)

    it("formats DogStatsD lines", function()
        assert.equal(
            "trajectory.gen_ai.client.inference.usage.cost:0.3|c|#a:b,c:d",
            dogstatsd.dogstatsd_line({
                name = "trajectory.gen_ai.client.inference.usage.cost",
                type = "count",
                value = 0.3,
                tags = { "a:b", "c:d" },
            })
        )
        assert.equal(
            "gen_ai.client.inference.duration:12|d",
            dogstatsd.dogstatsd_line({
                name = "gen_ai.client.inference.duration",
                type = "distribution",
                value = 12,
                tags = {},
            })
        )
    end)

    it("normalizes tag values one character per code point", function()
        assert.equal("ada_example.com", dogstatsd.datadog_tag_value("Ada@Example.com"))
        assert.equal("a__b", dogstatsd.datadog_tag_value("A  B"))
        assert.equal("_", dogstatsd.datadog_tag_value("\195\169"))
        assert.equal("e_", dogstatsd.datadog_tag_value("e\204\129"))
        assert.equal("", dogstatsd.datadog_tag_value(""))
        assert.equal("a-b_c:d.e/f", dogstatsd.datadog_tag_value("a-b_c:d.e/f"))
    end)
end)
