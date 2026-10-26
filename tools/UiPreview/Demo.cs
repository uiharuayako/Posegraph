using System.Numerics;
using System.Reflection;
using HarmonyLib;
using Posegraph;
using Posegraph.Backend;
using Posegraph.Library;
using Posegraph.UI;

namespace UiPreview;

/// <summary>
/// 教程截图用的演示环境（--demo）：图库与识别缓存放在临时目录（不碰用户自己的图库，也不会把个人图片拍进公开教程），
/// 后端路径显示成通用的一键包路径，GPose 里有几个假角色。识别要本机后端在线（默认端口）。
/// </summary>
public static class Demo
{
    public static bool Enabled { get; set; }
    public static string Root { get; } = Path.Combine(Path.GetTempPath(), "PosegraphUiPreview", "demo");
    public const string BackendDir = @"D:\Posegraph-Backend\backend";
    public const string Python = @"D:\Posegraph-Backend\python\python.exe";

    public static readonly (int Index, string Name)[] Actors = [(201, "Aria Moonshade"), (202, "Kael Ironwood"), (203, "Lumi Starwhisper")];

    private static readonly BindingFlags Field = BindingFlags.NonPublic | BindingFlags.Instance;

    /// <summary>示例图复制进临时图库（教程目录里的 CC0 照片）。</summary>
    public static void Setup(App app, string samples)
    {
        var images = Path.Combine(Root, "Images");
        var cache = Path.Combine(Root, "Cache");
        // 图库每次重建；识别缓存保留（第二种语言、重拍时不用再跑模型）
        if (Directory.Exists(images)) Directory.Delete(images, true);
        Directory.CreateDirectory(Path.Combine(images, Posegraph.Loc.T("动作参考")));
        Directory.CreateDirectory(cache);
        foreach (var file in Directory.GetFiles(samples, "sample-*.jpg"))
        {
            var name = Path.GetFileNameWithoutExtension(file)["sample-".Length..];
            File.Copy(file, Path.Combine(images, name + ".jpg"));
        }
        File.Copy(Path.Combine(images, "kalaripayattu.jpg"), Path.Combine(images, Posegraph.Loc.T("动作参考"), "kalaripayattu.jpg"));
        var lib = app.Main.Library;
        typeof(ImageLibrary).GetField("<Root>k__BackingField", Field)!.SetValue(lib, images);
        typeof(ImageLibrary).GetField("<CacheDir>k__BackingField", Field)!.SetValue(lib, cache);
        app.Config.BackendDir = BackendDir;
        app.Config.BackendPython = Python;
        app.Config.LibraryFolder = images;
    }

    public static string Image(string name) => Path.Combine(Root, "Images", name + ".jpg");

    /// <summary>画帧直到后端在线（主窗口每 2 秒查一次）。</summary>
    public static void WaitBackend(App app, Action draw, int seconds = 30)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (!app.Main.BackendOnline && DateTime.UtcNow < until)
        {
            Host.Current.Frame(draw);
            Thread.Sleep(100);
        }
        if (!app.Main.BackendOnline) throw new InvalidOperationException("后端不在线（默认端口 42430）");
    }

    /// <summary>主窗口打开一张图，画帧直到识别和重定向都完成。</summary>
    public static void Open(App app, string image, Action draw, int seconds = 180)
    {
        WaitBackend(app, draw);
        typeof(MainWindow).GetField("_result", Field)!.SetValue(app.Main, null);
        app.Main.OpenImage(image);
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (DateTime.UtcNow < until)
        {
            Host.Current.Frame(draw);
            Thread.Sleep(50);
            var recognizing = (bool)typeof(MainWindow).GetField("_recognizing", Field)!.GetValue(app.Main)!;
            var busy = (bool)typeof(MainWindow).GetField("_rtBusy", Field)!.GetValue(app.Main)!;
            if (!recognizing && !busy && typeof(MainWindow).GetField("_result", Field)!.GetValue(app.Main) is not null) break;
        }
        for (var i = 0; i < 6; i++) Host.Current.Frame(draw);   // 纹理换上、布局稳定
    }

    /// <summary>角色列表里给角色绑图（绑定按角色身份存在 SplitWindow 里）。</summary>
    public static void Bind(SplitWindow split, int actor, string image)
    {
        var bindings = (Dictionary<string, string>)typeof(SplitWindow).GetField("_bindings", Field)!.GetValue(split)!;
        var name = Actors.First(a => a.Index == actor).Name;
        bindings[$"{actor}:{name}"] = image;
    }
}

// 演示时后端路径校验通过、显示已安装的后端版本（路径是假的）
[HarmonyPatch(typeof(BackendProcess), nameof(BackendProcess.Validate))]
internal static class ValidatePatch
{
    private static bool Prefix(ref string? __result)
    {
        if (!Demo.Enabled) return true;
        __result = null;
        return false;
    }
}

[HarmonyPatch(typeof(BackendUpdater), nameof(BackendUpdater.Installed))]
internal static class InstalledPatch
{
    private static bool Prefix(ref BackendUpdater.VersionInfo? __result)
    {
        if (!Demo.Enabled) return true;
        __result = new BackendUpdater.VersionInfo(BackendUpdater.PluginTag[1..], 1);
        return false;
    }
}

[HarmonyPatch(typeof(BackendUpdater), nameof(BackendUpdater.PackageRuntime))]
internal static class RuntimePatch
{
    private static bool Prefix(ref int? __result)
    {
        if (!Demo.Enabled) return true;
        __result = 1;
        return false;
    }
}

[HarmonyPatch(typeof(BackendUpdater), nameof(BackendUpdater.NeedsUpdate))]
internal static class NeedsUpdatePatch
{
    private static bool Prefix(ref bool __result)
    {
        if (!Demo.Enabled) return true;
        __result = false;
        return false;
    }
}
