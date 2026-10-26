using System.Net;
using System.Net.Sockets;
using System.Text;

internal sealed record MockResponse(int StatusCode, string Body);
internal sealed record CapturedRequest(string Method, string Path, string Body, string? Language, string? ContentType, string? Query = null);

/// <summary>Real loopback HTTP, so production serialization, handlers and error parsing are exercised.</summary>
internal sealed class LocalBackend : IDisposable
{
    private readonly HttpListener _listener = new();
    public int Port { get; }
    public Task<CapturedRequest[]> Requests { get; }

    public LocalBackend(params MockResponse[] responses)
    {
        // Obtain an available OS-assigned port; immediately bind the HTTP listener.
        using (var probe = new TcpListener(IPAddress.Loopback, 0))
        {
            probe.Start();
            Port = ((IPEndPoint)probe.LocalEndpoint).Port;
        }
        _listener.Prefixes.Add($"http://127.0.0.1:{Port}/");
        _listener.Start();
        Requests = ServeAsync(responses);
    }

    private async Task<CapturedRequest[]> ServeAsync(MockResponse[] responses)
    {
        var requests = new List<CapturedRequest>();
        foreach (var response in responses)
        {
            var context = await _listener.GetContextAsync().WaitAsync(TimeSpan.FromSeconds(10));
            using var reader = new StreamReader(context.Request.InputStream);
            var body = await reader.ReadToEndAsync();
            requests.Add(new CapturedRequest(context.Request.HttpMethod, context.Request.Url!.AbsolutePath,
                body, context.Request.Headers["Accept-Language"], context.Request.ContentType, context.Request.Url.Query));
            var bytes = Encoding.UTF8.GetBytes(response.Body);
            context.Response.StatusCode = response.StatusCode;
            context.Response.ContentType = "application/json; charset=utf-8";
            context.Response.ContentLength64 = bytes.Length;
            await context.Response.OutputStream.WriteAsync(bytes);
            context.Response.Close();
        }
        return requests.ToArray();
    }

    public void Dispose() => _listener.Close();
}
