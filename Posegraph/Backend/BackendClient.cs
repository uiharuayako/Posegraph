using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;

namespace Posegraph.Backend;

/// <summary>识别后端（backend）的 HTTP 客户端。接口见 backend/src/posegraph/app.py。</summary>
public sealed class BackendClient(Func<int> port) : IDisposable
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    // 识别首次要加载模型，可能超过半分钟
    private readonly HttpClient _http = new(new LanguageHandler()) { Timeout = TimeSpan.FromSeconds(180) };

    /// <summary>每个请求带上界面语言，后端返回的提示 / 错误信息按它翻译。</summary>
    private sealed class LanguageHandler() : DelegatingHandler(new HttpClientHandler())
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            request.Headers.AcceptLanguage.Clear();
            request.Headers.AcceptLanguage.Add(new StringWithQualityHeaderValue(Loc.Lang));
            return base.SendAsync(request, ct);
        }
    }

    private string Url(string path) => $"http://127.0.0.1:{port()}{path}";

    public async Task<Health?> HealthAsync(CancellationToken ct = default)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(TimeSpan.FromSeconds(2));
        return await _http.GetFromJsonAsync<Health>(Url("/health"), Json, cts.Token);
    }

    public Task LoadModelAsync() => PostAsync("/model/load", null);

    public Task UnloadModelAsync() => PostAsync("/model/unload", null);

    public Task ShutdownAsync() => PostAsync("/shutdown", null);

    /// <summary>运行时修改设备与释放策略（设备改动在下次加载时生效）。</summary>
    public Task ConfigureAsync(Configuration c) => PostAsync("/model/config",
        new Dictionary<string, object> { ["device"] = c.BackendDevice, ["release_after"] = c.ReleaseAfter, ["mhr_on_cpu"] = c.MhrOnCpu });

    /// <summary>每批预识别开始前同步并行上限，设备和释放策略保持不变。</summary>
    public async Task ConfigureRecognitionAsync(int parallelism)
    {
        var limit = Math.Clamp(parallelism, 1, 8);
        using var resp = await _http.PostAsJsonAsync(Url("/model/config"), new { recognition_parallelism = limit }, Json);
        var body = await ReadAsync<JsonElement>(resp);
        var confirmed = body.ValueKind == JsonValueKind.Object && body.TryGetProperty("model", out var model) &&
            model.ValueKind == JsonValueKind.Object && model.TryGetProperty("config", out var config) &&
            config.ValueKind == JsonValueKind.Object && config.TryGetProperty("recognitionParallelism", out var value) &&
            value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var actual) && actual == limit;
        // Older backends serialize inference and ignore this unknown config field.
        // Accept that only for the default; never present HTTP fan-out as real parallel inference.
        if (limit > 1 && !confirmed)
            throw new InvalidOperationException(Loc.T("当前后端不支持并行识别，请更新并重启后端"));
    }

    /// <summary>hold：批次内先保留模型，所有批次结束后再按释放策略卸载。</summary>
    public async Task<SessionInfo> CreateSessionAsync(byte[] image, string name, bool hold = false, CancellationToken ct = default)
    {
        using var content = new ByteArrayContent(image);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        using var resp = await _http.PostAsync(Url($"/sessions?name={Uri.EscapeDataString(name)}{(hold ? "&hold=true" : "")}"), content, ct);
        return (await ReadAsync<SessionInfo>(resp, ct))!;
    }

    public Task<byte[]> SessionImageAsync(string sid, CancellationToken ct = default) => _http.GetByteArrayAsync(Url($"/sessions/{sid}/image"), ct);

    /// <summary>会话存档（npz），存进图库的识别缓存。</summary>
    public async Task<byte[]> SessionArchiveAsync(string sid, CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(Url($"/sessions/{sid}/archive"), ct);
        await EnsureSuccessAsync(resp, ct);
        return await resp.Content.ReadAsByteArrayAsync(ct);
    }

    /// <summary>从识别缓存恢复会话（不加载模型）。</summary>
    public async Task<SessionInfo> RestoreSessionAsync(byte[] archive, string name, CancellationToken ct = default)
    {
        using var content = new ByteArrayContent(archive);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        using var resp = await _http.PostAsync(Url($"/sessions/archive?name={Uri.EscapeDataString(name)}"), content, ct);
        return (await ReadAsync<SessionInfo>(resp, ct))!;
    }

    public async Task DeleteSessionAsync(string sid, CancellationToken ct = default)
    {
        using var resp = await _http.DeleteAsync(Url($"/sessions/{sid}"), ct);
        await EnsureSuccessAsync(resp, ct);
    }

    public async Task<RetargetResult> RetargetAsync(string sid, int person, RetargetParams p, string? reference, string? race,
                                                    bool trustReference = false, bool useAux = true)
    {
        using var resp = await _http.PostAsJsonAsync(Url($"/sessions/{sid}/retarget"),
            new { person, @params = p, reference, race, overlay = true, stick = true, trust_reference = trustReference, use_aux = useAux }, Json);
        return (await ReadAsync<RetargetResult>(resp))!;
    }

    public async Task<CameraResult> CameraAsync(string sid, int person, RetargetParams p, int? actor, string? race, bool useAux = true)
    {
        using var resp = await _http.PostAsJsonAsync(Url($"/sessions/{sid}/camera"), new { person, @params = p, actor, race, use_aux = useAux }, Json);
        return (await ReadAsync<CameraResult>(resp))!;
    }

    /// <summary>给会话挂上辅助视角（各辅助图的会话，取第 0 人）；空数组为去掉。</summary>
    public async Task<AuxView[]> SetViewsAsync(string sid, IEnumerable<string> aux)
    {
        using var resp = await _http.PostAsJsonAsync(Url($"/sessions/{sid}/views"), new { aux = aux.ToArray() }, Json);
        return (await ReadAsync<ViewsResult>(resp))!.Views;
    }

    private async Task PostAsync(string path, object? body)
    {
        using var resp = await _http.PostAsJsonAsync(Url(path), body ?? new { }, Json);
        await ReadAsync<JsonElement>(resp);
    }

    /// <summary>非 2xx 时把后端 FastAPI 的 detail 作为异常信息抛出。</summary>
    private static async Task<T?> ReadAsync<T>(HttpResponseMessage resp, CancellationToken ct = default)
    {
        await EnsureSuccessAsync(resp, ct);
        return await resp.Content.ReadFromJsonAsync<T>(Json, ct);
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage resp, CancellationToken ct = default)
    {
        if (resp.IsSuccessStatusCode) return;
        var text = await resp.Content.ReadAsStringAsync(ct);
        string detail = text;
        try { detail = JsonDocument.Parse(text).RootElement.GetProperty("detail").ToString(); } catch { }
        throw new BackendException((int)resp.StatusCode, Loc.F("后端返回 {0}：{1}", (int)resp.StatusCode, detail));
    }

    public void Dispose() => _http.Dispose();
}

/// <summary>后端返回非 2xx。StatusCode 404 = 会话不存在（后端重启或会话被挤出），可从识别缓存恢复。</summary>
public sealed class BackendException(int statusCode, string message) : InvalidOperationException(message)
{
    public int StatusCode { get; } = statusCode;
}
