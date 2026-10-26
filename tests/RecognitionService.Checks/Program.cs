using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Posegraph.Backend;
using Posegraph.Library;

await BoundsAndFailure();
await DuplicateContentAndForeground();
await StopAndDispose();
await ConfigurationAndArchiveFailure();
await CorruptCacheAndConcurrentSaves();
await BatchHoldAndAuxiliaryViews();
await NestedBatchLifecycle();
Console.WriteLine("7 library recognition integration scenarios passed (real service/library/client; loopback HTTP).");

static void Check(bool value, string message)
{
    if (!value) throw new InvalidOperationException(message);
}
static async Task Idle(RecognitionService service)
{
    while (service.Busy) await service.Completion.WaitAsync(TimeSpan.FromSeconds(10));
}
static async Task BoundsAndFailure()
{
    foreach (var (requested, limit) in new[] { (1, 1), (3, 3), (99, 8) })
    {
        using var f = new Fixture(requested);
        f.Service.Completed += (_, _) => throw new InvalidOperationException("subscriber failure");
        var paths = Enumerable.Range(0, limit + 2).Select(i => f.Image("image" + i, i)).ToArray();
        f.Service.Enqueue(paths);
        var running = new List<Job>();
        for (var i = 0; i < limit; i++) running.Add(await f.Backend.Next());
        Check(f.Backend.Created == limit && f.Service.Pending == 2, "scheduler exceeded its configured cap");
        f.Parallelism = 1; // current batch retains the original snapshot
        running[^1].Completion.SetResult(false);
        var next = await f.Backend.Next();
        next.Completion.SetResult(true);
        var last = await f.Backend.Next();
        last.Completion.SetResult(true);
        foreach (var job in running.Take(limit - 1)) job.Completion.SetResult(true);
        await Idle(f.Service);
        Check(f.Backend.Configured.SequenceEqual([limit]) && f.Backend.Peak <= limit,
              "concurrency was not bounded or changed mid-batch");
        Check(paths.Count(p => f.Service.Status(p)?.State == RecognitionState.Done) == paths.Length - 1 &&
              paths.Count(p => f.Service.Status(p)?.State == RecognitionState.Failed) == 1,
              "failure stopped later jobs or changed result ownership");
        Check(f.Backend.Deleted.Count == paths.Length - 1, "successful intermediate sessions leaked");
    }
    Console.WriteLine("PASS bounded concurrency, snapshot, out-of-order failure, subscriber isolation, cleanup");
}
static async Task DuplicateContentAndForeground()
{
    using var f = new Fixture(3);
    var a = f.Image("same-a", 42);
    var b = f.Image("same-b", 42);
    f.Service.Enqueue([a, b]);
    var job = await f.Backend.Next();
    var foreground = f.Service.OpenAsync(b);
    Check(!foreground.IsCompleted, "foreground bypassed the per-content archive gate");
    job.Completion.SetResult(true);
    var selected = await foreground.WaitAsync(TimeSpan.FromSeconds(10));
    await Idle(f.Service);
    Check(f.Backend.Created == 1 && f.Backend.Restored == 1 && selected.Info.Device is null,
          "same-content paths or foreground selection repeated inference instead of archive restoration");
    Check(!f.Backend.Deleted.Contains(selected.Info.SessionId) && f.Backend.Deleted.Contains(job.Id),
          "batch cleanup deleted the foreground selected session");
    var restored = await f.Service.RestoreAsync(selected.Sha1, "after-backend-restart");
    Check(restored?.Device is null && f.Backend.Created == 1 && f.Backend.Restored == 2,
          "current-session recovery repeated inference instead of restoring the library archive");
    Check(Directory.GetFiles(f.Library.CacheDir, "*.npz").Length == 1 &&
          Directory.GetFiles(f.Library.CacheDir, "*.tmp").Length == 0,
          "same-hash cache writers collided or left temporary archives");
    Console.WriteLine("PASS duplicate-content cache sharing and foreground selection during batch");
}
static async Task StopAndDispose()
{
    foreach (var dispose in new[] { false, true })
    {
        using var f = new Fixture(2);
        var paths = Enumerable.Range(0, 5).Select(i => f.Image("stop" + i, i)).ToArray();
        f.Service.Enqueue(paths);
        var a = await f.Backend.Next(); var b = await f.Backend.Next();
        var callbacks = 0;
        f.Service.Completed += (_, _) => callbacks++;
        if (dispose) f.Service.Dispose(); else f.Service.CancelPending();
        Check(f.Service.Pending == 0 && !f.Service.Completion.IsCompleted, "stop aborted live work or retained queued work");
        a.Completion.SetResult(true); b.Completion.SetResult(true);
        await f.Service.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Check(f.Backend.Created == 2 && f.Backend.Deleted.Count == 2 &&
              paths.Skip(2).All(p => f.Service.Status(p) is null), "stop/dispose lost cleanup or started skipped items");
        Check(callbacks == (dispose ? 0 : 2), "dispose delivered a late UI callback or stop suppressed live results");
    }
    Console.WriteLine("PASS stop/dispose skips pending work, drains in-flight archive/delete lifecycle");
}
static async Task ConfigurationAndArchiveFailure()
{
    using (var f = new Fixture(2))
    {
        f.Backend.ConfigError = true;
        var paths = new[] { f.Image("a", 1), f.Image("b", 2) };
        f.Service.Enqueue(paths); await Idle(f.Service);
        Check(f.Backend.Created == 0 && paths.All(p => f.Service.Status(p)?.State == RecognitionState.Failed),
              "failed capability negotiation started inference");
    }
    using (var f = new Fixture(1))
    {
        f.Backend.ArchiveError = true;
        var path = f.Image("archive-error", 3);
        f.Service.Enqueue([path]);
        (await f.Backend.Next()).Completion.SetResult(true);
        await Idle(f.Service);
        Check(f.Service.Status(path)?.State == RecognitionState.Failed && f.Backend.Deleted.Count == 1,
              "archive failure leaked the already-created session");
    }
    Console.WriteLine("PASS backend capability failure and archive failure cleanup");
}
static async Task CorruptCacheAndConcurrentSaves()
{
    using var f = new Fixture(1);
    var path = f.Image("bad-cache", 12);
    var hash = f.Library.RememberHash(path, File.ReadAllBytes(path));
    f.Library.WriteCache(hash, [0]);
    f.Backend.RejectArchive = true;
    var opening = f.Service.OpenAsync(path);
    (await f.Backend.Next()).Completion.SetResult(true);
    var opened = await opening.WaitAsync(TimeSpan.FromSeconds(10));
    Check(f.Backend.Restored == 1 && f.Backend.Created == 1 && opened.Info.Device == "cpu",
          "status-400 archive fallback did not regenerate the corrupted cache");
    var paths = Enumerable.Range(0, 32).Select(i => f.Image("save" + i, i)).ToArray();
    await Task.WhenAll(paths.Select(p => Task.Run(() => { f.Library.RememberHash(p, File.ReadAllBytes(p)); f.Library.Save(); })));
    var reopened = new ImageLibrary(f.Directory, f.Directory);
    Check(paths.All(p => reopened.KnownHash(p) is not null), "concurrent index saves lost newer hash entries");
    Console.WriteLine("PASS corrupted-cache repair and serialized, lossless library index saves");
}

static async Task BatchHoldAndAuxiliaryViews()
{
    using var f = new Fixture(3) { ReleaseImmediately = true };
    var main = f.Image("main", 30);
    var auxiliary = f.Image("side", 31);
    var hash = f.Library.RememberHash(main, File.ReadAllBytes(main));
    var attached = f.Library.AddAux(hash, [auxiliary]).Single();
    f.Service.Enqueue([main]);
    var mainJob = await f.Backend.Next();
    Check(mainJob.Name == "main" && mainJob.Hold, "main batch request did not retain the model");
    mainJob.Completion.SetResult(true);
    var auxJob = await f.Backend.Next();
    Check(auxJob.Name == "side" && auxJob.Hold, "auxiliary pre-recognition was skipped or lost its model hold");
    auxJob.Completion.SetResult(true);
    await Idle(f.Service);
    Check(f.Backend.Created == 2 && f.Backend.Unloaded == 1 && f.Backend.Deleted.Count == 2 &&
          f.Library.HasCache(hash) && f.Library.HasCache(f.Library.Hash(attached)),
          "batch did not cache main/auxiliary results, clean sessions and unload once");
    var views = await f.Service.AttachAuxAsync("foreground-main", hash);
    Check(views.Length == 1 && views[0].Path == attached && f.Service.ViewOf(attached)?.View == "left" &&
          f.Backend.Restored == 1 && f.Backend.Created == 2 && f.Backend.Unloaded == 1 && f.Backend.Deleted.Count == 3,
          "auxiliary fusion lost view metadata, repeated inference or leaked the cached temporary session");
    f.Backend.ViewsError = true;
    try { await f.Service.AttachAuxAsync("foreground-main", hash); throw new Exception("Expected view failure"); }
    catch (BackendException error) when (error.StatusCode == 500) { }
    Check(f.Backend.Deleted.Count == 4 && f.Backend.Unloaded == 1,
          "failed auxiliary fusion leaked a restored session or unloaded a cache-only batch");
    Console.WriteLine("PASS 0.4 batch hold, auxiliary pre-recognition/cache/view metadata and failure cleanup");
}

static async Task NestedBatchLifecycle()
{
    using var f = new Fixture(1) { ReleaseImmediately = true };
    var outer = f.Service.Batch();
    var inner = f.Service.Batch();
    var path = f.Image("nested", 50);
    var opening = f.Service.OpenAsync(path);
    var job = await f.Backend.Next();
    Check(job.Hold, "nested batch did not hold its foreground inference");
    job.Completion.SetResult(true);
    await opening;
    await inner.DisposeAsync();
    Check(f.Backend.Unloaded == 0 && !f.Service.Completion.IsCompleted, "inner batch released the model or was not tracked");
    await outer.DisposeAsync();
    await outer.DisposeAsync();
    Check(f.Backend.Unloaded == 1, "nested/idempotent batch disposal did not release exactly once");
    await using (f.Service.Batch()) await f.Service.OpenAsync(path);
    Check(f.Backend.Unloaded == 1, "cache-only batch unnecessarily unloaded the model");
    var direct = f.Service.OpenAsync(f.Image("single", 51));
    var directJob = await f.Backend.Next();
    Check(!directJob.Hold, "ordinary foreground request acquired a batch hold");
    directJob.Completion.SetResult(true);
    await direct;
    Console.WriteLine("PASS nested/idempotent batch lifetime, cache-only batches and ordinary foreground requests");
}

sealed class Fixture : IDisposable
{
    public readonly string Directory = Path.Combine(Path.GetTempPath(), "Posegraph-LibraryChecks-" + Guid.NewGuid().ToString("N"));
    public readonly ScriptedBackend Backend = new();
    public readonly BackendClient Client;
    public readonly ImageLibrary Library;
    public readonly RecognitionService Service;
    public int Parallelism;
    public bool ReleaseImmediately;
    public Fixture(int parallelism)
    {
        System.IO.Directory.CreateDirectory(Directory);
        Library = new ImageLibrary(Directory, Directory);
        Client = new BackendClient(() => Backend.Port);
        Parallelism = parallelism;
        Service = new RecognitionService(Library, Client, () => Parallelism, () => ReleaseImmediately);
    }
    public string Image(string name, int marker)
    {
        var path = Path.Combine(Directory, name + ".png");
        File.WriteAllBytes(path, [(byte)marker]); return path;
    }
    public void Dispose()
    {
        Service.Dispose();
        Service.Completion.GetAwaiter().GetResult();
        Client.Dispose(); Backend.Dispose();
        var full = Path.GetFullPath(Directory);
        CheckPath(full);
        System.IO.Directory.Delete(full, true);
    }
    private static void CheckPath(string path)
    {
        var root = Path.GetFullPath(Path.GetTempPath()).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        if (!path.StartsWith(root, StringComparison.OrdinalIgnoreCase) || !Path.GetFileName(path).StartsWith("Posegraph-LibraryChecks-"))
            throw new InvalidOperationException("Unexpected fixture directory");
    }
}
sealed record Job(string Id, TaskCompletionSource<bool> Completion, string Name, bool Hold);
sealed class ScriptedBackend : IDisposable
{
    private readonly HttpListener _listener = new();
    private readonly Channel<Job> _jobs = Channel.CreateUnbounded<Job>();
    public readonly ConcurrentQueue<string> Deleted = new();
    public readonly ConcurrentQueue<int> Configured = new();
    public int Created, Restored, Peak, Unloaded;
    private int _active, _ids;
    public bool ConfigError, ArchiveError, RejectArchive, ViewsError;
    public int Port { get; }
    public ScriptedBackend()
    {
        using var probe = new TcpListener(IPAddress.Loopback, 0); probe.Start();
        Port = ((IPEndPoint)probe.LocalEndpoint).Port; probe.Stop();
        _listener.Prefixes.Add($"http://127.0.0.1:{Port}/"); _listener.Start();
        _ = Listen();
    }
    public async Task<Job> Next() => await _jobs.Reader.ReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10));
    private async Task Listen()
    {
        try { while (_listener.IsListening) _ = Handle(await _listener.GetContextAsync()); }
        catch (HttpListenerException) { } catch (ObjectDisposedException) { }
    }
    private async Task Handle(HttpListenerContext context)
    {
        var req = context.Request;
        var path = req.Url!.AbsolutePath;
        var status = 200;
        string body;
        if (path == "/model/config")
        {
            using var doc = await JsonDocument.ParseAsync(req.InputStream);
            var limit = doc.RootElement.GetProperty("recognition_parallelism").GetInt32();
            Configured.Enqueue(limit); status = ConfigError ? 422 : 200;
            body = JsonSerializer.Serialize(new { model = new { config = new { recognitionParallelism = limit } } });
        }
        else if (path == "/model/unload")
        {
            Interlocked.Increment(ref Unloaded);
            body = "{}";
        }
        else if (path.EndsWith("/views"))
        {
            using var doc = await JsonDocument.ParseAsync(req.InputStream);
            var count = doc.RootElement.GetProperty("aux").GetArrayLength();
            status = ViewsError ? 500 : 200;
            body = ViewsError ? "{\"detail\":\"view failure\"}" : JsonSerializer.Serialize(new
                { views = Enumerable.Range(0, count).Select(_ => new { yaw = 90, view = "left", share = .5, conflict = 0, worst = "", worstDeg = 0 }).ToArray() });
        }
        else if (path == "/sessions/archive")
        {
            Interlocked.Increment(ref Restored);
            status = RejectArchive ? 400 : 200;
            body = status == 400 ? "{\"detail\":\"bad archive\"}" : Session("restored-" + Interlocked.Increment(ref _ids), req.QueryString["name"]!, null);
        }
        else if (path == "/sessions" && req.HttpMethod == "POST")
        {
            using var stream = new MemoryStream(); await req.InputStream.CopyToAsync(stream);
            var id = "created-" + Interlocked.Increment(ref _ids);
            Interlocked.Increment(ref Created);
            var active = Interlocked.Increment(ref _active);
            int previous; do { previous = Peak; } while (active > previous && Interlocked.CompareExchange(ref Peak, active, previous) != previous);
            var job = new Job(id, new(TaskCreationOptions.RunContinuationsAsynchronously), req.QueryString["name"]!, req.QueryString["hold"] == "true");
            await _jobs.Writer.WriteAsync(job);
            var ok = await job.Completion.Task; Interlocked.Decrement(ref _active);
            status = ok ? 200 : 500;
            body = ok ? Session(id, req.QueryString["name"]!, "cpu") : "{\"detail\":\"bad image\"}";
        }
        else if (path.EndsWith("/archive")) { status = ArchiveError ? 500 : 200; body = "archive-bytes"; }
        else if (req.HttpMethod == "DELETE") { Deleted.Enqueue(path.Split('/')[2]); body = "{\"ok\":true}"; }
        else { status = 404; body = "{}"; }
        var bytes = Encoding.UTF8.GetBytes(body);
        context.Response.StatusCode = status; context.Response.ContentType = "application/json";
        context.Response.ContentLength64 = bytes.Length;
        try { await context.Response.OutputStream.WriteAsync(bytes); } finally { context.Response.Close(); }
    }
    private static string Session(string sid, string name, string? device)
        => JsonSerializer.Serialize(new { sessionId = sid, name, imageSize = new[] { 1, 1 }, inferenceMs = 1, device, humans = Array.Empty<object>() });
    public void Dispose() { _listener.Close(); }
}
