using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Posegraph;
using Posegraph.Backend;

var checks = new (string Name, Func<Task> Run)[]
{
    ("Clamps values below one and posts only recognition_parallelism", ClampLowerBound),
    ("Clamps values above eight", ClampUpperBound),
    ("Accepts matching backend confirmation for each concurrency 1 through 8", MatchingConfirmation),
    ("Rejects missing parallelism confirmation above one", MissingConfirmation),
    ("Rejects mismatched parallelism confirmation above one", MismatchedConfirmation),
    ("Rejects malformed parallelism confirmation above one", MalformedConfirmation),
    ("Accepts default one on older backend without confirmation", DefaultCompatible),
    ("Propagates non-success HTTP status and backend detail", BackendError),
    ("Sends current Accept-Language on every configuration request", LanguageHeader),
    ("Preserves batch hold and escaped image names", BatchHold),
};

foreach (var (name, run) in checks)
{
    await run().WaitAsync(TimeSpan.FromSeconds(20));
    Console.WriteLine($"PASS {name}");
}
Console.WriteLine($"{checks.Length} BackendClient checks passed (production source linked, real loopback HTTP).");

static string Confirmation(int count)
    => JsonSerializer.Serialize(new { model = new { config = new { recognitionParallelism = count } } });

static async Task<(Exception? Error, CapturedRequest Request)> Exchange(int parallelism, string response, int status = 200)
{
    using var backend = new LocalBackend(new MockResponse(status, response));
    using var client = new BackendClient(() => backend.Port);
    Exception? error = null;
    try { await client.ConfigureRecognitionAsync(parallelism).WaitAsync(TimeSpan.FromSeconds(10)); }
    catch (Exception ex) { error = ex; }
    var request = (await backend.Requests)[0];
    Assert(request.Method == "POST" && request.Path == "/model/config", "Unexpected request method/path");
    Assert(request.ContentType?.StartsWith("application/json", StringComparison.OrdinalIgnoreCase) == true,
        "Configuration request was not JSON");
    Assert(request.Language == Loc.Lang, "Configuration request lost Accept-Language");
    return (error, request);
}

static void AssertPayload(CapturedRequest request, int expected)
{
    using var body = JsonDocument.Parse(request.Body);
    Assert(body.RootElement.GetProperty("recognition_parallelism").GetInt32() == expected,
        $"Expected serialized recognition_parallelism={expected}");
    Assert(body.RootElement.EnumerateObject().Count() == 1,
        "Recognition configuration unexpectedly overwrote other backend settings");
}

static void AssertUnsupported(Exception? error)
{
    Assert(error is InvalidOperationException && error.Message == "当前后端不支持并行识别，请更新并重启后端",
        $"Expected backend upgrade error, got {error?.ToString() ?? "success"}");
}

static async Task ClampLowerBound()
{
    var (error, request) = await Exchange(int.MinValue, Confirmation(1));
    Assert(error is null, "Clamped default was rejected");
    AssertPayload(request, 1);
}

static async Task ClampUpperBound()
{
    var (error, request) = await Exchange(int.MaxValue, Confirmation(8));
    Assert(error is null, "Clamped maximum was rejected");
    AssertPayload(request, 8);
}

static async Task MatchingConfirmation()
{
    for (var count = 1; count <= 8; count++)
    {
        var (error, request) = await Exchange(count, Confirmation(count));
        Assert(error is null, $"Matching confirmation for {count} was rejected: {error}");
        AssertPayload(request, count);
    }
}

static async Task MissingConfirmation()
{
    foreach (var response in new[] { "{}", "{\"model\":{}}", "{\"model\":{\"config\":{}}}" })
    {
        var (error, request) = await Exchange(3, response);
        AssertUnsupported(error);
        AssertPayload(request, 3);
    }
}

static async Task MismatchedConfirmation()
{
    var (error, _) = await Exchange(4, Confirmation(1));
    AssertUnsupported(error);
}

static async Task MalformedConfirmation()
{
    foreach (var value in new[] { "\"3\"", "null", "true", "3.5" })
    {
        var (error, _) = await Exchange(3, "{\"model\":{\"config\":{\"recognitionParallelism\":" + value + "}}}");
        AssertUnsupported(error);
    }
}

static async Task DefaultCompatible()
{
    var (error, request) = await Exchange(1, "{\"model\":{\"config\":{}}}");
    Assert(error is null, $"Default sequential mode rejected older backend: {error}");
    AssertPayload(request, 1);
}

static async Task BackendError()
{
    var (error, _) = await Exchange(4, "{\"detail\":\"invalid recognition_parallelism\"}", 422);
    Assert(error is InvalidOperationException && error.Message.Contains("422") &&
        error.Message.Contains("invalid recognition_parallelism"), "Backend status/detail were not preserved");
}

static async Task LanguageHeader()
{
    using var backend = new LocalBackend(new MockResponse(200, Confirmation(1)), new MockResponse(200, Confirmation(2)));
    using var client = new BackendClient(() => backend.Port);
    var original = Loc.Lang;
    try
    {
        Loc.Lang = "zh";
        await client.ConfigureRecognitionAsync(1);
        Loc.Lang = "en";
        await client.ConfigureRecognitionAsync(2);
        var requests = await backend.Requests;
        Assert(requests[0].Language == "zh" && requests[1].Language == "en",
            "Accept-Language did not follow the current UI language on the same client");
    }
    finally { Loc.Lang = original; }
}

static async Task BatchHold()
{
    const string response = "{\"sessionId\":\"check\",\"name\":\"image\",\"imageSize\":[1,1],\"inferenceMs\":1,\"device\":\"cpu\",\"humans\":[]}";
    using var backend = new LocalBackend(new MockResponse(200, response), new MockResponse(200, response));
    using var client = new BackendClient(() => backend.Port);
    const string name = "中文 & image?";
    await client.CreateSessionAsync([1], name, hold: true);
    await client.CreateSessionAsync([2], name);
    var requests = await backend.Requests;
    Assert(requests.All(request => request.Method == "POST" && request.Path == "/sessions" &&
        request.Query!.Contains("name=" + Uri.EscapeDataString(name)) && request.Language == Loc.Lang),
        "Session request lost its name or language");
    Assert(requests[0].Query!.EndsWith("&hold=true") && !requests[1].Query!.Contains("hold="),
        "Batch hold was lost or applied to an ordinary foreground request");
}

static void Assert(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}
