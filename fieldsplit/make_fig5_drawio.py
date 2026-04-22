from pathlib import Path

OUT = Path("/home/chris/Documents/chagas/paper_figures/fig5_architecture.drawio")
BACKBONE_BLOCKS = [
    ("Block 1", 16, "~650 × 650"),
    ("Block 2", 32, "~325 × 325"),
    ("Block 3", 64, "~162 × 162"),
    ("Block 4", 128, "~81 × 81"),
    ("Block 5", 256, "~40 × 40"),
    ("Block 6", 512, "~20 × 20"),
    ("Block 7", 1024, "~10 × 10"),
]


def main() -> None:
    col_backbone_x = 80
    col_lateral_x = 370
    col_pyramid_x = 620
    col_head_x = 900
    block_w = 220
    block_h = 72
    block_vgap = 30
    top_pad = 180
    lateral_w = 160
    lateral_h = 44
    pyramid_w = 180
    pyramid_h = 72
    cells = []
    next_id = [10]

    def cid():
        v = next_id[0]
        next_id[0] += 1
        return f"c{v}"

    def box(
        x,
        y,
        w,
        h,
        label,
        fill="#dae8fc",
        stroke="#6c8ebf",
        font_size=12,
        bold=False,
        rounded=False,
    ):
        style = f"{('rounded=1;arcSize=15;' if rounded else 'rounded=0;')}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize={font_size};{('fontStyle=1;' if bold else '')}verticalAlign=middle;align=center;"
        i = cid()
        label_escaped = (
            label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        cells.append(
            f'<mxCell id="{i}" value="{label_escaped}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return i

    def arrow(src, dst, label="", style_extra="", dashed=False):
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=classic;strokeColor=#555555;fontSize=11;"
            + (f"dashed=1;" if dashed else "")
            + style_extra
        )
        i = cid()
        lab = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        cells.append(
            f'<mxCell id="{i}" value="{lab}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )
        return i

    box(
        col_backbone_x,
        30,
        col_head_x + 220 - col_backbone_x,
        40,
        "Feature-pyramid classifier for T. cruzi",
        fill="none",
        stroke="none",
        font_size=18,
        bold=True,
    )
    box(
        col_backbone_x,
        70,
        col_head_x + 220 - col_backbone_x,
        28,
        "Seven-block VGG-style backbone, top-down pyramid with 1×1 laterals, single classification head on p5",
        fill="none",
        stroke="none",
        font_size=13,
    )
    input_id = box(
        col_backbone_x,
        110,
        block_w,
        56,
        "Input image\n1,300 × 1,300 × 3",
        fill="#f5f5f5",
        stroke="#666666",
        bold=True,
    )
    backbone_ids = []
    for i, (name, ch, spatial) in enumerate(BACKBONE_BLOCKS):
        y = top_pad + i * (block_h + block_vgap)
        shade = [
            "#dae8fc",
            "#cde0fb",
            "#b9d1f5",
            "#a4c1ef",
            "#8fb2e9",
            "#7ba3e3",
            "#6694dd",
        ][i]
        label = f"{name}\n3×3 conv × 3\n{ch} ch · {spatial}"
        bid = box(col_backbone_x, y, block_w, block_h, label, fill=shade)
        backbone_ids.append(bid)
    arrow(input_id, backbone_ids[0])
    for a, b in zip(backbone_ids[:-1], backbone_ids[1:]):
        arrow(a, b)
    lateral_ids = []
    for i, bid in enumerate(backbone_ids):
        y = top_pad + i * (block_h + block_vgap) + (block_h - lateral_h) // 2
        lid = box(
            col_lateral_x,
            y,
            lateral_w,
            lateral_h,
            "1×1 conv → 256 ch",
            fill="#fff2cc",
            stroke="#d6b656",
            font_size=11,
        )
        lateral_ids.append(lid)
        arrow(bid, lid)
    pyramid_ids = []
    pyramid_labels = [f"p{i + 1}" for i in range(7)]
    for i, plabel in enumerate(pyramid_labels):
        y = top_pad + i * (block_h + block_vgap)
        pid = box(
            col_pyramid_x,
            y,
            pyramid_w,
            pyramid_h,
            f"{plabel}\n256 channels",
            fill="#d5e8d4",
            stroke="#82b366",
            bold=plabel == "p5",
        )
        pyramid_ids.append(pid)
    plus_size = 28
    plus_ids = [None] * 7

    def circle(x, y, w, h, label, fill="#e1d5e7", stroke="#9673a6"):
        style = f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=16;fontStyle=1;verticalAlign=middle;align=center;"
        i = cid()
        cells.append(
            f'<mxCell id="{i}" value="{label}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return i

    for i in range(6):
        py_y = top_pad + i * (block_h + block_vgap)
        cx = col_pyramid_x - plus_size - 18
        cy = py_y + (pyramid_h - plus_size) // 2
        plus_ids[i] = circle(cx, cy, plus_size, plus_size, "+")
    for i, (lid, pid) in enumerate(zip(lateral_ids, pyramid_ids)):
        if plus_ids[i] is not None:
            arrow(lid, plus_ids[i])
            arrow(plus_ids[i], pid)
        else:
            arrow(lid, pid)
    for i in range(6):
        arrow(
            pyramid_ids[i + 1],
            plus_ids[i],
            style_extra="strokeColor=#82b366;strokeWidth=2;",
        )
    p5_id = pyramid_ids[4]
    head_y_top = top_pad + 4 * (block_h + block_vgap)
    head_boxes = [
        ("3×3 conv → 1 ch", "#f8cecc", "#b85450"),
        ("Global average pool", "#f8cecc", "#b85450"),
        ("Fully connected (1)", "#f8cecc", "#b85450"),
        ("Sigmoid", "#f8cecc", "#b85450"),
    ]
    head_box_w = 170
    head_box_h = 48
    head_gap = 14
    head_ids = []
    x = col_head_x
    y = head_y_top + (block_h - head_box_h) // 2
    for label, fill, stroke in head_boxes:
        hid = box(
            x, y, head_box_w, head_box_h, label, fill=fill, stroke=stroke, font_size=11
        )
        head_ids.append(hid)
        x += head_box_w + head_gap
    arrow(p5_id, head_ids[0])
    for a, b in zip(head_ids[:-1], head_ids[1:]):
        arrow(a, b)
    out_x = x
    out_y = head_y_top + (block_h - head_box_h) // 2
    out_id = box(
        out_x,
        out_y,
        220,
        head_box_h,
        "Parasite probability\n(0 to 1)",
        fill="#e1d5e7",
        stroke="#9673a6",
        bold=True,
        font_size=12,
    )
    arrow(head_ids[-1], out_id)
    legend_y = top_pad + len(BACKBONE_BLOCKS) * (block_h + block_vgap) + 30
    legend_w = col_head_x + 220 - col_backbone_x
    box(
        col_backbone_x,
        legend_y,
        legend_w,
        64,
        "Each backbone block: three 3×3 convolutions (BN + ReLU) followed by 2×2 max-pooling (stride 2) and dropout (rate 0.5).\nTop-down green arrows between pyramid levels: 2× nearest-neighbor upsample of the coarser level, element-wise summed with the lateral projection (⊕).",
        fill="none",
        stroke="none",
        font_size=12,
    )
    box(
        col_backbone_x,
        legend_y + 76,
        legend_w,
        28,
        "Blue = VGG backbone block   |   Yellow = 1×1 lateral convolution (→ 256 ch)   |   Green = pyramid feature map   |   Red = classification head",
        fill="none",
        stroke="none",
        font_size=12,
    )
    cell_xml = "\n".join(cells)
    doc = f'<mxfile host="app.diagrams.net" modified="2026-04-21T00:00:00.000Z" agent="chagas-paper" version="24.0.0" type="device">\n  <diagram id="fig5-fpn" name="Fig 5 — FPN architecture">\n    <mxGraphModel dx="1800" dy="1400" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1400" math="0" shadow="0">\n      <root>\n        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n{cell_xml}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    print(f"Wrote {OUT}")
    print(
        f"Open at https://app.diagrams.net or in draw.io desktop; edit to taste, then:"
    )
    print(f"  File -> Export As -> Advanced -> DPI 300 -> format TIFF")
    print(f"  Save as paper_figures/plos_submission/Fig5.tif")


if __name__ == "__main__":
    main()
