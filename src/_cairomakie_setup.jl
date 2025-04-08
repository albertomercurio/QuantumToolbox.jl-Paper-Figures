# Use the command `\the\textwidth` in LaTeX to get the width of the line in points
# Use the command
# ```
# \the\textwidth
# ```
# to get the font size in points
const mm_to_pt = 2.83465
const plot_figsize_width_single_column_pt = 231.84843
const plot_figsize_width_pt = 483.69687
const plot_figsize_height_pt = plot_figsize_width_pt * 0.6 # this may vary

#=
510.0pt for APS textwidth
246.0pt for APS linewidth
483.69687pt for Quantum textwidth
231.84843pt for Quantum linewidth
180mm for Nature Communications wide figure
88mm for Nature Communications narrow figure
=#

const plot_labelsize = 9
const _my_theme_Cairo = Theme(
    fontsize = plot_labelsize,
    figure_padding = (1,7,1,5),
    size = (plot_figsize_width_pt, plot_figsize_height_pt),
    Axis = (
        spinewidth=0.7,
        xgridvisible=false,
        ygridvisible=false,
        xtickwidth=0.75,
        ytickwidth=0.75,
        xminortickwidth=0.5,
        yminortickwidth=0.5,
        xticksize=3,
        yticksize=3,
        xminorticksize=1.5,
        yminorticksize=1.5,
        xlabelpadding=1,
        ylabelpadding=1,
        xticklabelsize=plot_labelsize,
        yticklabelsize=plot_labelsize,
    ),
    Legend = (
        merge=true,
        framevisible=false,
        patchsize=(15,2),
    ),
    Lines = (
        linewidth=1.6,
    ),
)
const my_theme_Cairo = merge(_my_theme_Cairo, theme_latexfonts())

CairoMakie.set_theme!(my_theme_Cairo)
CairoMakie.activate!(type = "svg", pt_per_unit = 1.25)
CairoMakie.enable_only_mime!(MIME"image/svg+xml"())

# helper function to get the position inside the figure
function posFig(ax, x, y)
    o = ax.scene.viewport[].origin
    return Makie.project(ax.scene, Point2f(x, y)) + o
end

function InsetAxis(fig, pos, ax, x_zoom, y_zoom; width=Relative(0.3), height=Relative(0.5), halign=0.97, valign=0.2, backgroundcolor=(:white, 1.0))
    inset_ax = Axis(pos; width=width, height=height, halign=halign, valign=valign, backgroundcolor=backgroundcolor)

    x1_zoom, x2_zoom = x_zoom
    y1_zoom, y2_zoom = y_zoom

    ax_origin = inset_ax.scene.viewport[].origin
    ax_widths = inset_ax.scene.viewport[].widths
    point_bl = posFig(ax, x1_zoom, y1_zoom)
    point_br = posFig(ax, x2_zoom, y1_zoom)

    poly!(ax, [Point2f(x1_zoom, y1_zoom), Point2f(x2_zoom, y1_zoom), Point2f(x2_zoom, y2_zoom), Point2f(x1_zoom, y2_zoom)], color=:transparent, strokecolor=:black, strokewidth=0.8)

    poly!(fig.scene, [Point2f(ax_origin[1], ax_origin[2]+ax_widths[2]), Point2f(point_bl[1], point_bl[2]), Point2f(point_br[1], point_br[2]), Point2f(ax_origin[1]+ax_widths[1], ax_origin[2]+ax_widths[2])], color=:transparent, strokecolor=:grey, strokewidth=0.8)

    return inset_ax
end