using System.Numerics;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.ImGuiFileDialog;
using Dalamud.Interface.Utility;
using Dalamud.Plugin.Services;
using Posegraph.Backend;
using Posegraph.Library;

namespace Posegraph.UI;

/// <summary>
/// 辅助视角菜单（图库右键的「辅助视角」子菜单、主窗口工具栏共用）。同一动作其他角度的图挂在主图上，不在图库网格里显示；
/// 可以从文件复制、粘贴剪切板、把图库里选中的其他图挂上来。每张可以手动标左 / 右 / 背面，只用于提示和检查，融合本身不需要。
/// </summary>
public sealed class AuxViewsMenu : IDisposable
{
    private static readonly Vector4 Green = PosegraphTheme.Green;
    private static readonly Vector4 Yellow = PosegraphTheme.Yellow;
    private static readonly Vector4 Red = PosegraphTheme.Red;

    private static readonly string[] ManualViews = ["left", "right", "back"];

    private readonly ImageLibrary _library;
    private readonly RecognitionService _recognition;
    private readonly ITextureProvider _textures;
    private readonly ITextureReadbackProvider _readback;
    private readonly ThumbnailCache _thumbs;
    private readonly FileDialogManager _dialog = new();
    private readonly Configuration _config;

    public AuxViewsMenu(Configuration config, ImageLibrary library, RecognitionService recognition, ITextureProvider textures,
                        ITextureReadbackProvider readback, IPluginLog log)
    {
        _config = config;
        _library = library;
        _recognition = recognition;
        _textures = textures;
        _readback = readback;
        _thumbs = new ThumbnailCache(textures, log, (int)(200 * ImGuiHelpers.GlobalScale));
    }

    public void Dispose() => _thumbs.Dispose();

    /// <summary>文件选择框（由 Plugin 每帧画一次，菜单关了也要能弹出）。</summary>
    public void DrawDialogs() => _dialog.Draw();

    public static string ViewLabel(string view) => view switch
    {
        "left" => Loc.T("左侧"),
        "right" => Loc.T("右侧"),
        "back" => Loc.T("背面"),
        "same" => Loc.T("几乎同角度"),
        _ => view,
    };

    /// <summary>一张辅助图的视角说明：手动指定的优先，否则用上次融合时识别出的；还没融合过显示「未识别」。</summary>
    public string Describe(string sha1, string path)
    {
        var manual = _library.AuxLabel(sha1, path);
        var detected = _recognition.ViewOf(path);
        if (manual is not null)
            return detected is not null && detected.View != manual && detected.View != "same"
                ? Loc.F("{0}（手动；识别像{1}，可能左右反了）", ViewLabel(manual), ViewLabel(detected.View))
                : Loc.F("{0}（手动）", ViewLabel(manual));
        return detected is null ? Loc.T("未识别") : $"{ViewLabel(detected.View)} {detected.Yaw:+0;-0}°";
    }

    // multiview.LIMBS 里的骨段
    private static readonly Dictionary<string, string> LimbNames = new()
    {
        ["c_spine1"] = "腰背", ["l_upleg"] = "左大腿", ["l_lowleg"] = "左小腿", ["r_upleg"] = "右大腿", ["r_lowleg"] = "右小腿",
        ["l_uparm"] = "左上臂", ["l_lowarm"] = "左前臂", ["r_uparm"] = "右上臂", ["r_lowarm"] = "右前臂",
    };

    /// <summary>融合效果的提示（主窗口状态、菜单里显示）；没有问题返回 null。</summary>
    public static string? Warning(AuxView v)
        => v.View == "same" ? Loc.T("和主图几乎同一角度，补不了深度，帮助不大")
         : v.Conflict > 25 ? Loc.F("和主图动作对不上（平均差 {0:0}°，{1}差 {2:0}°），AI 生成的图可能改了动作", v.Conflict,
                                   LimbNames.TryGetValue(v.Worst, out var n) ? Loc.T(n) : v.Worst, v.WorstDeg)
         : null;

    /// <summary>
    /// 画在一个已打开的菜单 / 弹窗里。main：主图；others：图库里同时选中的其他图（可以移过来作为辅助视角）。
    /// status：在调用方窗口的状态栏显示结果。
    /// </summary>
    public void Draw(string main, IReadOnlyList<string> others, Action<string, Vector4> status)
    {
        string sha1;
        try
        {
            sha1 = _library.KnownHash(main) ?? _library.Hash(main);
        }
        catch (Exception ex)
        {
            ImGui.TextColored(Red, ex.Message);
            return;
        }
        ImGui.TextDisabled(Loc.T("同一动作其他角度的图，识别时自动融合，补单张图猜不准的前后深度"));
        ImGui.TextDisabled(Loc.T("最有用的是约 90° 的侧面；背面、和主图角度相近的帮助不大"));
        ImGui.Separator();

        var aux = _library.AuxImages(sha1);
        if (aux.Length == 0) ImGui.TextDisabled(Loc.T("（还没有辅助视角图）"));
        foreach (var path in aux)
        {
            var view = _recognition.ViewOf(path);
            var warn = view is not null && Warning(view) is not null;
            if (!ImGui.BeginMenu($"{Path.GetFileNameWithoutExtension(path)}　{Describe(sha1, path)}{(warn ? "  (!)" : "")}##{path}")) continue;
            if (_thumbs.Get(path, out _) is { } wrap)
            {
                var side = 200 * ImGuiHelpers.GlobalScale;
                var fit = Math.Min(side / wrap.Width, side / wrap.Height);
                ImGui.Image(wrap.Handle, new Vector2(wrap.Width, wrap.Height) * fit);
            }
            if (view is not null)
            {
                ImGui.TextDisabled(Loc.F("机位：{0} {1:+0;-0}°，主要骨段采纳 {2:0}%", ViewLabel(view.View), view.Yaw, view.Share * 100));
                if (Warning(view) is { } w) ImGui.TextColored(Yellow, w);
            }
            else ImGui.TextDisabled(Loc.T("主图在主窗口打开时识别并融合"));
            ImGui.Separator();
            var manual = _library.AuxLabel(sha1, path);
            if (ImGui.MenuItem(Loc.T("视角：自动判断"), manual is null, true)) _library.SetAuxLabel(sha1, path, null);
            foreach (var key in ManualViews)
                if (ImGui.MenuItem(Loc.F("视角：{0}", ViewLabel(key)), manual == key, true)) _library.SetAuxLabel(sha1, path, key);
            ImGui.Separator();
            if (ImGui.MenuItem(Loc.T("在资源管理器中显示"))) ShowInExplorer(path);
            if (ImGui.MenuItem(Loc.T("移除（进回收站）")))
            {
                Act(status, () =>
                {
                    _library.RemoveAux(sha1, path);
                    return Loc.F("已移除辅助视角：{0}", Path.GetFileName(path));
                });
                _thumbs.Forget(path);
            }
            ImGui.EndMenu();
        }

        ImGui.Separator();
        if (ImGui.MenuItem(Loc.T("从文件添加…"))) AddFromFiles(sha1, status);
        if (ImGui.MenuItem(Loc.T("粘贴剪切板里的图片"), false, _textures.HasClipboardImage())) Paste(sha1, status);
        var movable = others.Where(p => File.Exists(p) && !string.Equals(p, main, StringComparison.OrdinalIgnoreCase)).ToList();
        if (movable.Count > 0 && ImGui.MenuItem(Loc.F("把选中的另外 {0} 张挂上来（移出图库）", movable.Count)))
            Act(status, () => Loc.F("已挂上 {0} 张辅助视角", _library.AddAux(sha1, movable, move: true).Count));
        if (ImGui.IsItemHovered()) ImGui.SetTooltip(Loc.T("文件名为「主图名-1」「主图名-2」的图导入图库时会自动挂上"));
    }

    private void AddFromFiles(string sha1, Action<string, Vector4> status)
    {
        var dir = Directory.Exists(_config.LastImageDir) ? _config.LastImageDir : null;
        _dialog.OpenFileDialog(Loc.T("添加辅助视角图（复制）"), Loc.T("图片") + "{.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff,.avif}", (ok, paths) =>
        {
            if (!ok || paths.Count == 0) return;
            Act(status, () =>
            {
                var added = _library.AddAux(sha1, paths);
                return added.Count == paths.Count ? Loc.F("已添加 {0} 张辅助视角", added.Count)
                                                  : Loc.F("已添加 {0} 张辅助视角（{1} 张和主图或已有的相同，已跳过）", added.Count, paths.Count - added.Count);
            });
        }, 0, dir, true);
    }

    private void Paste(string sha1, Action<string, Vector4> status)
        => _ = Task.Run(async () =>
        {
            try
            {
                var png = await ClipboardImage.ReadPngAsync(_textures, _readback);
                if (png is null)
                {
                    status(Loc.T("剪切板里没有图片"), Yellow);
                    return;
                }
                status(_library.AddAuxPng(sha1, png) is { } path ? Loc.F("已添加辅助视角：{0}", Path.GetFileName(path))
                                                                 : Loc.T("和主图或已有的辅助视角相同，已跳过"), Green);
            }
            catch (Exception ex)
            {
                status(Loc.F("读取剪切板失败：{0}", ex.Message), Red);
            }
        });

    /// <summary>文件操作放到后台（复制大图、算哈希），完成后报告状态。</summary>
    private static void Act(Action<string, Vector4> status, Func<string> action)
        => _ = Task.Run(() =>
        {
            try
            {
                status(action(), Green);
            }
            catch (Exception ex)
            {
                status(ex.Message, Red);
            }
        });

    private static void ShowInExplorer(string path)
    {
        try
        {
            System.Diagnostics.Process.Start("explorer.exe", $"/select,\"{path}\"");
        }
        catch
        {
            // 打不开资源管理器不影响使用
        }
    }
}
