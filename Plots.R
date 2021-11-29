library(ggplot2)

real <- read.csv('SpectraReal.csv',header=F)
args <- commandArgs(trailingOnly = TRUE)

if (args[1] == '20'){
  points <- read.csv('Spectra20.csv',header=F)
} else {
  points <- read.csv('Spectra100.csv',header=F)
}

predictions <- read.csv(args[2], header=F)

myplot <- ggplot() + geom_point(data=points, aes(x=V1, y=V2, color="Datos")) + geom_line(data=real, aes(x=V1, y=V2, color="Valores reales"))+ geom_line(data=predictions, aes(x=V1, y=V2, color="Aproximación obtenida"))+ scale_color_discrete('Curva') + xlab('x') + ylab('y') + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))
ggsave('plot.png')