cd(@__DIR__)

using Plots
using DelimitedFiles
using LaTeXStrings
using Measures

function scatter_style(xl,yl)
    scatter!(
        	ylabel=yl, xlabel=xl,
            grid = :off,
            box = :on,
            foreground_color_legend = nothing,
            fontfamily = "serif-roman",
            font="CMU Serif",
            xtickfontsize = 12,
            ytickfontsize = 12,
            xguidefontsize = 12,
            yguidefontsize = 12,
            thickness_scaling=1,
            legendfontsize=10,
            yguidefontrotation=0,
            #legend_font_pointsize=14,
            #legendtitlefontsize=14,
            markersize=1,
            legend=:topright,
            margin=5mm
        )
end


Nevents = 20
Ntime = 20 

df = [readdlm("data/shearwave_1_$i.dat") for i in 1:Ntime]

for id in 1:Nevents 
    df .+= [readdlm("data/shearwave_$id"*"_$i.dat") for i in 1:Ntime]
end 

df .*= 1.0/Nevents 



plot()

for i in 1:Ntime 
    plot!(df[i][2:end], label="t="*string(round(20*df[i][1],digits=3) ))
end 

scatter_style(L"y", L"\pi_x")


savefig("shear_relax.pdf")

