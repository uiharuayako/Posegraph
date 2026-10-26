using System.Numerics;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface;
using Dalamud.Interface.Components;
using Dalamud.Interface.DragDrop;
using Dalamud.Interface.Utility;
using Dalamud.Interface.Utility.Raii;
using Dalamud.Interface.Windowing;
using Dalamud.Plugin;
using Dalamud.Plugin.Services;
using Posegraph.Library;

namespace Posegraph.UI;

/// <summary>
/// 角色列表（拆分模式）：GPose 里的每个角色绑一张图（从图库拖进来），一起应用；某个角色要细调时「编辑」切到主窗口。
/// 每张图用它自己保存的修正（3D 页保存），挂了辅助视角就融合。绑定按角色身份记，只在本次 GPose 内有效（角色编号每次进 GPose 会变）。
/// 默认贴在主窗口右边。
/// </summary>
public sealed class SplitWindow : Window, IDisposable
{
    private static readonly Vector4 Green = PosegraphTheme.Green;
    private static readonly Vector4 Yellow = PosegraphTheme.Yellow;
    private static readonly Vector4 Red = PosegraphTheme.Red;
    private static readonly Vector4 Grey = PosegraphTheme.Grey;
    private static readonly Vector4 Blue = PosegraphTheme.Blue;

    private readonly Configuration _config;
    private readonly IDalamudPluginInterface _pi;
    private readonly MainWindow _main;
    private readonly LibraryWindow _libraryWindow;
    private readonly ImageLibrary _library;
    private readonly AuxViewsMenu _auxMenu;
    private readonly IDragDropManager _dragDrop;
    private readonly ThumbnailCache _thumbs;

    private readonly Dictionary<string, string> _bindings = new();                       // 角色身份 → 图片
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, (string Text, Vector4 Color)> _rowStatus = new();   // 角色身份 → 上次应用结果（后台写）
    private readonly Dictionary<int, string> _keys = new();                                // ObjectIndex → 角色身份（每秒刷新）
    private DateTime _lastKeys = DateTime.MinValue;
    private bool _wasGposing;
    private volatile bool _applying;
    private string _status = "";
    private Vector4 _statusColor = Grey;
    private string? _contextKey;

    public SplitWindow(Configuration config, IDalamudPluginInterface pi, MainWindow main, LibraryWindow libraryWindow,
                       ImageLibrary library, AuxViewsMenu auxMenu, ITextureProvider textures, IDragDropManager dragDrop, IPluginLog log)
        : base(Loc.T("Posegraph 角色列表") + "###PosegraphSplit")
    {
        _config = config;
        _pi = pi;
        _main = main;
        _libraryWindow = libraryWindow;
        _library = library;
        _auxMenu = auxMenu;
        _dragDrop = dragDrop;
        _thumbs = new ThumbnailCache(textures, log, (int)(64 * ImGuiHelpers.GlobalScale));
        Size = new Vector2(340, 420);
        SizeCondition = ImGuiCond.FirstUseEver;
        SizeConstraints = new WindowSizeConstraints { MinimumSize = new Vector2(280, 200), MaximumSize = new Vector2(2000, 4000) };
    }

    public void Dispose() => _thumbs.Dispose();

    private void SetStatus(string text, Vector4 color)
    {
        _status = text;
        _statusColor = color;
    }

    public override void PreDraw()
    {
        // 贴在主窗口右边；主窗口没打开（拿不到位置）时不强制
        if (_config.SplitDock && _main.IsOpen && _main.WindowSize.X > 0)
        {
            Position = _main.WindowPos + new Vector2(_main.WindowSize.X + 4, 0);
            PositionCondition = ImGuiCond.Always;
        }
        else
        {
            Position = null;
        }
        PosegraphTheme.Push();
    }

    public override void PostDraw() => PosegraphTheme.Pop();

    private void RefreshKeys()
    {
        if (_main.InGpose != _wasGposing)
        {
            // 进出 GPose：角色编号会重新分配，绑定作废
            _wasGposing = _main.InGpose;
            _bindings.Clear();
            _rowStatus.Clear();
            _keys.Clear();
        }
        if (DateTime.UtcNow - _lastKeys < TimeSpan.FromSeconds(1)) return;
        _lastKeys = DateTime.UtcNow;
        _keys.Clear();
        foreach (var (index, _) in _main.Actors)
            if (_main.Identify(index) is { } id) _keys[index] = id.Key;
    }

    public override void Draw()
    {
        RefreshKeys();
        DrawHeader();
        ImGui.Separator();
        var footer = ImGui.GetFrameHeightWithSpacing() * 2;
        using (var list = ImRaii.Child("##actors", new Vector2(0, -footer)))
            if (list) DrawActors();
        DrawFooter();
    }

    private void DrawHeader()
    {
        var dock = _config.SplitDock;
        if (ImGui.Checkbox(Loc.T("贴在主窗口右边"), ref dock))
        {
            _config.SplitDock = dock;
            _pi.SavePluginConfig(_config);
        }
        ImGui.SameLine();
        ImGuiComponents.HelpMarker(Loc.T("从图库把图拖到角色左边的格子里绑定，每个角色一张图。\n「全部应用」用每张图保存的修正（3D 页保存），没有就用当前预设；挂了辅助视角的自动融合。\n要细调某个角色：点「编辑」切到主窗口，调好后在 3D 页保存修正。\n绑定只在这次 GPose 里有效。"));
    }

    private void DrawActors()
    {
        if (!_main.InGpose)
        {
            ImGui.TextDisabled(Loc.T("进入 GPose 后列出角色"));
            return;
        }
        if (_main.Actors.Length == 0)
        {
            ImGui.TextDisabled(Loc.T("没有可摆姿势的角色（Brio 里新建或选中角色）"));
            return;
        }
        foreach (var (index, name) in _main.Actors)
        {
            if (!_keys.TryGetValue(index, out var key)) continue;
            using (ImRaii.PushId(key))
                DrawRow(index, name, key);
        }
    }

    private void DrawRow(int index, string name, string key)
    {
        var scale = ImGuiHelpers.GlobalScale;
        var slot = 56 * scale;
        var image = _bindings.GetValueOrDefault(key);
        if (image is not null && !File.Exists(image))
        {
            _bindings.Remove(key);   // 图被删了或改名了
            image = null;
        }

        // 左边的图片格子：拖放目标，右键菜单
        var pos = ImGui.GetCursorScreenPos();
        ImGui.InvisibleButton("##slot", new Vector2(slot, slot));
        var hovered = ImGui.IsItemHovered();
        var dl = ImGui.GetWindowDrawList();
        dl.AddRectFilled(pos, pos + new Vector2(slot, slot), ImGui.GetColorU32(hovered ? ImGuiCol.FrameBgHovered : ImGuiCol.FrameBg), 4);
        if (image is not null && _thumbs.Get(image, out _) is { } wrap)
        {
            var fit = Math.Min((slot - 4) / wrap.Width, (slot - 4) / wrap.Height);
            var sz = new Vector2(wrap.Width, wrap.Height) * fit;
            var tl = pos + (new Vector2(slot, slot) - sz) / 2;
            dl.AddImage(wrap.Handle, tl, tl + sz);
        }
        else if (image is null)
        {
            var plus = "+";
            var sz = ImGui.CalcTextSize(plus);
            dl.AddText(pos + (new Vector2(slot, slot) - sz) / 2, ImGui.GetColorU32(ImGuiCol.TextDisabled), plus);
        }
        if (hovered && !ImGui.IsMouseDragging(ImGuiMouseButton.Left))
            ImGui.SetTooltip(image is null ? Loc.T("从图库把图拖到这里绑定，右键更多") : Path.GetFileName(image) + Loc.T("\n双击在主窗口编辑，右键更多"));
        if (hovered && image is not null && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left)) _main.Edit(index, image);
        AcceptDrop(key);
        if (ImGui.IsItemClicked(ImGuiMouseButton.Right))
        {
            _contextKey = key;
            ImGui.OpenPopup("##slotMenu");
        }
        if (ImGui.BeginPopup("##slotMenu"))
        {
            DrawSlotMenu(index, key, image);
            ImGui.EndPopup();
        }

        // 右边：名字、图片、状态、按钮
        ImGui.SameLine();
        using (ImRaii.Group())
        {
            UiText.Fit(name, ImGui.GetStyle().Colors[(int)ImGuiCol.Text]);
            if (image is null) ImGui.TextDisabled(Loc.T("未绑定图片"));
            else
            {
                var aux = _library.KnownHash(image) is { } sha ? _library.AuxCount(sha) : 0;
                UiText.Fit(Path.GetFileNameWithoutExtension(image) + (aux > 0 ? Loc.F("　辅助视角 {0}", aux) : ""), ImGui.GetStyle().Colors[(int)ImGuiCol.TextDisabled]);
            }
            if (_rowStatus.TryGetValue(key, out var st)) UiText.Fit(st.Text, st.Color);
            using (ImRaii.Disabled(image is null || _applying))
            {
                if (ImGui.SmallButton(Loc.T("编辑"))) _main.Edit(index, image!);
                if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled)) ImGui.SetTooltip(Loc.T("切到主窗口编辑这个角色（3D 页保存的修正，「全部应用」时也会用）"));
                ImGui.SameLine();
                if (ImGui.SmallButton(Loc.T("应用"))) ApplyAll([(index, key, image!)]);
            }
        }
        ImGui.Separator();
    }

    private void DrawSlotMenu(int index, string key, string? image)
    {
        if (_main.CurrentImage is { } current && ImGui.MenuItem(Loc.F("用主窗口当前的图（{0}）", Path.GetFileNameWithoutExtension(current))))
            Bind(key, current);
        if (image is not null)
        {
            if (ImGui.MenuItem(Loc.T("在主窗口编辑"))) _main.Edit(index, image);
            if (ImGui.BeginMenu(Loc.T("辅助视角")))
            {
                _auxMenu.Draw(image, [], SetStatus);
                ImGui.EndMenu();
            }
            if (ImGui.MenuItem(Loc.T("解除绑定")))
            {
                _bindings.Remove(key);
                _rowStatus.TryRemove(key, out _);
            }
        }
    }

    /// <summary>从图库拖来的图（取第一张），或从系统拖进来的图片文件。</summary>
    private void AcceptDrop(string key)
    {
        if (ImGui.BeginDragDropTarget())
        {
            var payload = ImGui.AcceptDragDropPayload(LibraryWindow.InternalDragLabel, ImGuiDragDropFlags.None);
            if (!payload.IsNull && _libraryWindow.DraggedImages.FirstOrDefault() is { } path) Bind(key, path);
            ImGui.EndDragDropTarget();
        }
        if (_dragDrop.CreateImGuiTarget(LibraryWindow.ExternalDragLabel, out var files, out _)
            && files.FirstOrDefault(ImageLibrary.IsImage) is { } file)
            Bind(key, file);
    }

    private void Bind(string key, string image)
    {
        _bindings[key] = image;
        _rowStatus.TryRemove(key, out _);
    }

    private void DrawFooter()
    {
        var rows = _main.Actors.Where(a => _keys.ContainsKey(a.Index))
                        .Select(a => (a.Index, Key: _keys[a.Index], Image: _bindings.GetValueOrDefault(_keys[a.Index])))
                        .Where(r => r.Image is not null).Select(r => (r.Index, r.Key, r.Image!)).ToList();
        using (ImRaii.Disabled(rows.Count == 0 || _applying || !_main.InGpose || !_main.BackendOnline))
            if (ImGui.Button(_applying ? Loc.T("应用中…") : Loc.F("全部应用（{0} 个角色）", rows.Count)))
                ApplyAll(rows);
        if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
            ImGui.SetTooltip(_main.BackendOnline ? Loc.T("按顺序逐个识别（有缓存的不重跑）并应用") : Loc.T("后端离线：先在主窗口启动后端"));
        UiText.Fit(_status, _statusColor);
    }

    private void ApplyAll(List<(int Index, string Key, string Image)> rows)
    {
        if (_applying) return;
        _applying = true;
        _ = Task.Run(async () =>
        {
            var ok = 0;
            try
            {
                await using var batch = _main.Recognition.Batch();   // 所有角色的图（含辅助视角）只加载一次模型
                foreach (var (index, key, image) in rows)
                {
                    _rowStatus[key] = (Loc.T("识别并应用中…"), Yellow);
                    SetStatus(Loc.F("正在应用 {0}…", Path.GetFileNameWithoutExtension(image)), Yellow);
                    try
                    {
                        var error = await _main.ApplyToActorAsync(index, image);
                        _rowStatus[key] = error is null ? (Loc.T("已应用"), Green) : (error, Red);
                        if (error is null) ok++;
                    }
                    catch (Exception ex)
                    {
                        _rowStatus[key] = (ex.Message, Red);
                    }
                }
                SetStatus(ok == rows.Count ? Loc.F("已应用 {0} 个角色", ok) : Loc.F("已应用 {0} 个，{1} 个失败", ok, rows.Count - ok),
                          ok == rows.Count ? Green : Red);
            }
            finally
            {
                _applying = false;
            }
        });
    }
}
