#File MedianEffectModel.R
# Analyze Yonetani & Theorell data with Median Effect Model
# Chou & Talalay, J Biol Chem 1977, Table IV and V for correct data
# Chou & Talalay, Eur J Biochem 1981, Table 1 and 2 for data with error
#   and Median Effect Model analysis
# Chou & Talalay "Quantitative analysis of dose-effect relationships:
#   the combined effects of multiple drugs or enzyme inhibitors."
#   Advances in enzyme regulation 1984, for the combination index
#
# Version 0.4, 2 january 2017 by Tjeerd Dijkstra
# Tested with R 3.3 under OS X 10.11.6 on MacPro 2013
setwd("/Users/Yuliya/Downloads/ChouTalalayAnalysis")
library(readr); library(tibble); library(ggplot2); library(dplyr)
# library(drc); library(minpack.lm); library(nlmrt); library(pracma)

# 36 observations of 3 variables: Conc.ADP, Conc.ADPr, FracInhib
yone.df1 <- read_tsv("YonetaniData1.tsv", col_types = "ddd", comment = "#")
# 36 observations of 3 variables: Conc.ADP, Conc.Phen, FracInhib
yone.df2 <- read_tsv("YonetaniData2.tsv", col_types = "ddd", comment = "#")
Dose1 <- unique(yone.df1$Conc.ADP); N.dose1 <- length(Dose1)
Dose2 <- unique(yone.df1$Conc.ADPr); N.dose2 <- length(Dose2)
Dose3 <- unique(yone.df2$Conc.Phen); N.dose3 <- length(Dose3)

# plot-ready response surface with log dose levels as row and column names
log_dose1 <- log10(Dose1)
log_dose1[1] <- log_dose1[2] - mean(diff(log_dose1[2:N.dose1]))
log_dose2 <- log10(Dose2)
log_dose2[1] <- log_dose2[2] - mean(diff(log_dose2[2:N.dose2]))
surf.mat1 <- matrix(yone.df1$FracInhib, N.dose1, N.dose2)
rownames(surf.mat1) <- log_dose1; colnames(surf.mat1) <- log_dose2
# second dataset shares ADP with first
log_dose3 <- log10(Dose3)
log_dose3[1] <- log_dose3[2] - mean(diff(log_dose3[2:N.dose3]))
surf.mat2 <- matrix(yone.df2$FracInhib, N.dose1, N.dose3)
rownames(surf.mat2) <- log_dose1; colnames(surf.mat2) <- log_dose3

# contour plot of response surface, note that rows and columns of the response matrix
# map to x and y (respectively) in the contour plot
library(RColorBrewer); contour.levels <- seq(from = 1, to = 0, by = -0.1)
countour.colors <- rev(brewer.pal(length(contour.levels), "RdYlBu"))
old.par <- par(mfrow = c(2, 1))
contour(as.numeric(rownames(surf.mat1)), as.numeric(colnames(surf.mat1)),
        surf.mat1, levels = contour.levels, col = countour.colors,
        xlab = sprintf("[ADP] [muM]"), ylab = sprintf("[ADPr] [muM]"),
        main = "Chou & Talalay 1981 Table 1"); grid()
contour(as.numeric(rownames(surf.mat2)), as.numeric(colnames(surf.mat2)),
        surf.mat2, levels = contour.levels, col = countour.colors,
        xlab = sprintf("[ADP] [muM]"), ylab = sprintf("[Phen] [muM]"),
        main = "Chou & Talalay 1981 Table 2"); grid()
par(old.par)

# contour plot of response surface using ggplot
# labeled contour function for ggplot by Hiroaki Yutani
StatContourLabel <- ggproto("StatContourLabel", Stat,
  default_aes = aes(label = ..level..),
  compute_group = function(..., digits = 1) {
    StatContour$compute_group(...) %>% group_by(level) %>%
      summarise(x = nth(x, round(n() / 2)), y = nth(y, round(n() / 2))) %>%
      mutate(level = signif(level, digits = digits)) 
  }
)

geom_contour_label <- function(mapping = NULL, data = NULL, position = "identity",
                    na.rm = FALSE, show.legend = NA, inherit.aes = TRUE, digits = 1, ...)
{ layer(data = data, mapping = mapping, stat = StatContourLabel, geom = "text",
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
    params = list(na.rm  = na.rm, digits = digits,  ...))
}

# multiple plot function by Winston Chang
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  
  if (is.null(layout)) {
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    for (i in 1:numPlots) {
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

contour.levels <- seq(from = 1, to = 0, by = -0.1)
p1 <- ggplot(yone.df1, aes(x = Conc.ADP, y = Conc.ADPr, z = FracInhib)) +
        geom_contour(aes(color = ..level..)) + geom_contour_label() +
        labs(x = "[ADP] [muM]", y = "[ADPr] [muM]",
             title = "Chou & Talalay 1981 Table 1")
p2 <- ggplot(yone.df2, aes(x = Conc.ADP, y = Conc.Phen, z = FracInhib)) +
        geom_contour(aes(color = ..level..)) + geom_contour_label() +
        labs(x = "[ADP] [muM]", y = "[Phen] [muM]",
              title = "Chou & Talalay 1981 Table 2")
multiplot(p1, p2, layout = matrix(1:2, nrow = 2))

# Contour plots <---



# recreate figures 4 and 6 of Chou & Talalay 1981 (fig 2 and 4 of CT1984)
mono.plus.diag <- c(surf.mat1[, 1], surf.mat1[1, ], diag(surf.mat1),
                    surf.mat2[, 1], surf.mat2[1, ], diag(surf.mat2))
LogDose <- c(log_dose2, log_dose1, log10(10^log_dose1 + 10^log_dose2),
             log_dose3, log_dose1, log10(10^log_dose1 + 10^log_dose3))
Dose <- c(Dose2, Dose1, Dose1 + Dose2, Dose3, Dose1, Dose1 + Dose3)
ChouTalalay <- c(rep("I1 ADPr", N.dose2), rep("I2 ADP 4", N.dose1), 
                 rep("I1+I2 190:1", N.dose1), rep("I1 Phen", N.dose2), 
                 rep("I2 ADP 6", N.dose1), rep("I1+I2 17.4:1", N.dose1))
CT.level <- c("I1 ADPr", "I2 ADP 4", "I1+I2 190:1", "I1 Phen", "I2 ADP 6", "I1+I2 17.4:1")
Figure <- c(rep("Figure 4", 3*N.dose1), rep("Figure 6", 3*N.dose1))
median.effect.df <- data_frame(LogMedEff = log10((1 - mono.plus.diag)/mono.plus.diag),
                                LogDose = LogDose, Dose = Dose,
                                Mixture = factor(ChouTalalay, CT.level), Figure = factor(Figure))
print(ggplot(median.effect.df[Dose > 0, ], aes(x = LogDose, y = LogMedEff, color = Mixture)) +
        geom_point() + geom_smooth(method = lm) + facet_grid(Figure ~ .) +
        labs(x = "log10 concentration [muM]", y = "log10 effect ratio []",
             title = "Chou & Talalay 1981"))

# linear model analysis of Chou and Talalay 1981
N.tot <-length(CT.level)
fit.res.df <- data.frame(I50.CT = c(156.1, 1.657, 107.0, 36.81, 1.656, 9.116),
                         IC50.me = rep(NA, N.tot),
                         slope.CT = c(0.968, 1.043, 1.004, 1.303, 1.187, 1.742),
                         slope.me = rep(NA, N.tot),
                         R.CT = c(0.9988, 0.9996, 0.9997, 0.9982, 0.9842, 0.9999),
                         R.me = rep(NA, N.tot))
row.names(fit.res.df) <- CT.level
for (i.tot in 1:N.tot){
  tmp.df <- median.effect.df[median.effect.df$Mixture == CT.level[i.tot] &
                                median.effect.df$Dose > 0, ]
  lm.out <- lm(LogMedEff ~ LogDose, data = tmp.df)
  fit.res.df$IC50.me[i.tot] <- 10^(-lm.out$coefficients[1]/lm.out$coefficients[2])
  fit.res.df$slope.me[i.tot] <- lm.out$coefficients[2]
  fit.res.df$R.me[i.tot] <- sqrt(summary(lm.out)$r.squared)
}
cat(sprintf("Chou & Talalay analysis with correct doses\n"))
print(fit.res.df, digits = 4)

# linear model analysis copying the swap of Chou and Talalay 1981
mono.diag.swap <- c(surf.mat1[1, ], surf.mat1[, 1], diag(surf.mat1),
                    surf.mat2[1, ], surf.mat2[, 1], diag(surf.mat2))
median.effect.df$LogMedEffSwapped <- log10((1 - mono.diag.swap)/mono.diag.swap)
for (i.tot in 1:N.tot){
  tmp.df <- median.effect.df[median.effect.df$Mixture == CT.level[i.tot] &
                               median.effect.df$Dose > 0, ]
  lm.out <- lm(LogMedEffSwapped ~ LogDose, data = tmp.df)
  fit.res.df$IC50.me[i.tot] <- 10^(-lm.out$coefficients[1]/lm.out$coefficients[2])
  fit.res.df$slope.me[i.tot] <- lm.out$coefficients[2]
  fit.res.df$R.me[i.tot] <- sqrt(summary(lm.out)$r.squared)
}
cat(sprintf("\nChou & Talalay analysis with swapped doses\n"))
print(fit.res.df, digits = 4)

# combination index analysis as on page 37 (bottom) and 38 (top) Chou&Talalay 1984
frac.eff <- seq(from = 0.01, to = 0.99, length.out = 40)
x.equiv <- function(frac.eff, I50, slope) { I50*(frac.eff/(1-frac.eff))^(1/slope) }

x.1.equiv <- x.equiv(frac.eff, fit.res.df$I50.CT[1], fit.res.df$slope.CT[1])
x.2.equiv <- x.equiv(frac.eff, fit.res.df$I50.CT[2], fit.res.df$slope.CT[2])
x.12.equiv <- x.equiv(frac.eff, fit.res.df$I50.CT[3], fit.res.df$slope.CT[3])
dose.ratio <- mean(Dose1[Dose1 > 0]/(Dose1[Dose1 > 0] + Dose2[Dose2 > 0]))
CI.fig3 <- x.12.equiv*(1-dose.ratio)/x.1.equiv + x.12.equiv*dose.ratio/x.2.equiv

x.1.equiv <- x.equiv(frac.eff, fit.res.df$I50.CT[4], fit.res.df$slope.CT[4])
x.2.equiv <- x.equiv(frac.eff, fit.res.df$I50.CT[5], fit.res.df$slope.CT[5])
x.12.equiv <- x.equiv(frac.eff, fit.res.df$I50.CT[6], fit.res.df$slope.CT[6])
dose.ratio <- mean(Dose1[Dose1 > 0]/(Dose1[Dose1 > 0] + Dose3[Dose3 > 0]))
CI.fig5 <- x.12.equiv*(1-dose.ratio)/x.1.equiv + x.12.equiv*dose.ratio/x.2.equiv

tmp.df <- data.frame(frac.eff = rep(frac.eff, 2), CI = c(CI.fig3, CI.fig5),
                     Figure = c(rep("Figure 3", length(frac.eff)),
                                rep("Figure 5", length(frac.eff))) )
print(ggplot(tmp.df, aes(x = frac.eff, y = CI)) +
        geom_line() + geom_hline(yintercept = 1) + facet_grid(Figure ~ .) +
        labs(x = "fractional effect []", y = "combination index []",
             title = "Chou & Talalay 1984"))

write.csv(tmp.df, "ChouTalalay.csv", row.names = FALSE)
