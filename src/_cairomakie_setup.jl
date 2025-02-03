# Use the command `\the\textwidth` in LaTeX to get the width of the line in points
# Use the command
# ```
# \the\textwidth
# ```
# to get the font size in points
const mm_to_pt = 2.83465
const plot_figsize_width_pt = 180 * mm_to_pt
const plot_figsize_height_pt = plot_figsize_width_pt * 0.6 # this may vary

#=
510.0pt for APS textwidth
246.0pt for APS linewidth
180mm for Nature Communications wide figure
88mm for Nature Communications narrow figure
=#

const plot_labelsize = 8
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