using Dalamud.Bindings.ImGui;
using SkiaSharp;

namespace UiPreview;

/// <summary>
/// 把 ImGui 的绘制数据软件光栅化到 SkiaSharp 画布（不需要显卡窗口）。
/// 纯色三角形（三个 UV 相同，采样字体图集的白点）Skia 求不出纹理映射会整个跳过，所以单独按顶点颜色画。
/// </summary>
public sealed class SkiaRenderer
{
    private readonly Dictionary<ulong, SKImage> _textures = new();
    private ulong _next = 1;

    public ulong Register(SKImage image)
    {
        var id = _next++;
        _textures[id] = image;
        return id;
    }

    public void Unregister(ulong id)
    {
        if (_textures.Remove(id, out var image)) image.Dispose();
    }

    public SKImage? Get(ulong id) => _textures.GetValueOrDefault(id);

    public unsafe void Render(ImDrawDataPtr data, SKCanvas canvas)
    {
        var lists = data.Handle->CmdLists;
        var origin = data.DisplayPos;
        for (var l = 0; l < data.CmdListsCount; l++)
        {
            var list = lists[l];
            var vtx = list->VtxBuffer;
            var idx = list->IdxBuffer;
            var cmds = list->CmdBuffer;
            for (var c = 0; c < cmds.Size; c++)
            {
                var cmd = cmds.Data[c];
                if (cmd.ElemCount == 0 || !_textures.TryGetValue(cmd.TextureId.Handle, out var tex)) continue;
                var solidPos = new List<SKPoint>();
                var solidCol = new List<SKColor>();
                var texPos = new List<SKPoint>();
                var texUv = new List<SKPoint>();
                var texCol = new List<SKColor>();
                var tri = new ImDrawVert[3];
                for (var k = 0; k < cmd.ElemCount; k += 3)
                {
                    for (var j = 0; j < 3; j++) tri[j] = vtx.Data[idx.Data[cmd.IdxOffset + k + j] + cmd.VtxOffset];
                    // 纹理坐标共线（面积为 0）时 Skia 求不出映射会跳过：纯色（三点相同）和整数粗细的抗锯齿线
                    // （纹理坐标在图集的同一行上）都按顶点颜色画
                    var a = tri[1].Uv - tri[0].Uv;
                    var b = tri[2].Uv - tri[0].Uv;
                    var solid = MathF.Abs(a.X * b.Y - a.Y * b.X) < 1e-9f;
                    foreach (var v in tri)
                    {
                        var p = new SKPoint(v.Pos.X - origin.X, v.Pos.Y - origin.Y);
                        var color = new SKColor((byte)v.Col, (byte)(v.Col >> 8), (byte)(v.Col >> 16), (byte)(v.Col >> 24));   // ABGR
                        if (solid)
                        {
                            solidPos.Add(p);
                            solidCol.Add(color);
                        }
                        else
                        {
                            texPos.Add(p);
                            texUv.Add(new SKPoint(v.Uv.X * tex.Width, v.Uv.Y * tex.Height));
                            texCol.Add(color);
                        }
                    }
                }
                canvas.Save();
                canvas.ClipRect(new SKRect(cmd.ClipRect.X - origin.X, cmd.ClipRect.Y - origin.Y, cmd.ClipRect.Z - origin.X, cmd.ClipRect.W - origin.Y));
                if (solidPos.Count > 0)
                {
                    using var paint = new SKPaint();
                    using var verts = SKVertices.CreateCopy(SKVertexMode.Triangles, solidPos.ToArray(), null, solidCol.ToArray());
                    canvas.DrawVertices(verts, SKBlendMode.Dst, paint);
                }
                if (texPos.Count > 0)
                {
                    using var shader = tex.ToShader(SKShaderTileMode.Clamp, SKShaderTileMode.Clamp, new SKSamplingOptions(SKFilterMode.Linear));
                    using var paint = new SKPaint { Shader = shader };
                    using var verts = SKVertices.CreateCopy(SKVertexMode.Triangles, texPos.ToArray(), texUv.ToArray(), texCol.ToArray());
                    canvas.DrawVertices(verts, SKBlendMode.Modulate, paint);
                }
                canvas.Restore();
            }
        }
    }
}
