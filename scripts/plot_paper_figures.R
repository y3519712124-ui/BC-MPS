# Reproducible vector figures for the fixed-epoch evidence; no generated data.
invisible(Sys.setlocale('LC_CTYPE', 'English_United States.utf8'))
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(svglite); library(ragg)})
dat <- read.csv('artifacts/paper_results/flat.csv')
grid <- read.csv('artifacts/paper_results/diagnostics/summary.csv')
dir.create('reproduced_figures', showWarnings=FALSE, recursive=TRUE)
stopifnot(nrow(dat) == 72, length(unique(dat$run)) == 9, nrow(grid) == 40)
ink <- '#26323D'
palette <- c(Fused='#168A9A', Adaptive='#D99532', 'HSI-only'='#4B9A70', 'LiDAR-only'='#8971B1')
label <- c(fused='Fused', adaptive='Adaptive', hsi='HSI-only', lidar='LiDAR-only')
dat$Predictor <- factor(label[dat$predictor], levels=names(palette))
dat$Rule <- factor(ifelse(dat$calibration == 'global_conformal', 'Global', 'BC-MPS'), levels=c('Global', 'BC-MPS'))
theme_pub <- theme_minimal(base_size=9, base_family='Arial') +
  theme(text=element_text(colour=ink), panel.grid.minor=element_blank(),
        panel.grid.major=element_line(colour='#E4E9EC', linewidth=.3),
        plot.title=element_text(size=10, face='bold'), legend.position='bottom',
        legend.title=element_blank(), plot.margin=margin(5,8,5,5))
agg <- function(metric) {
  out <- list()
  for (p in names(palette)) for (r in c('Global','BC-MPS')) {
    x <- dat[dat$Predictor==p & dat$Rule==r, metric]
    out[[length(out)+1]] <- data.frame(Predictor=p, Rule=r, value=mean(x), sd=sqrt(mean((x-mean(x))^2)))
  }
  d <- do.call(rbind,out)
  d$Predictor <- factor(d$Predictor, levels=rev(names(palette)))
  d$Rule <- factor(d$Rule, levels=c('Global','BC-MPS'))
  d
}
coverage_panel <- function(metric,title,show_legend=FALSE) {
  d <- agg(metric); d$value <- 100*d$value; d$sd <- 100*d$sd
  dodge <- position_dodge(width=.5)
  ggplot(d,aes(value,Predictor,colour=Predictor,shape=Rule,group=Rule)) +
    geom_vline(xintercept=90,linetype='dashed',colour='#75818A',linewidth=.4) +
    geom_errorbar(aes(xmin=value-sd,xmax=value+sd),orientation='y',width=.15,position=dodge,linewidth=.45) +
    geom_point(size=2.3,stroke=.6,position=dodge) +
    scale_colour_manual(values=palette,breaks=names(palette)) + scale_shape_manual(values=c(Global=1,'BC-MPS'=16)) +
    guides(colour=if(show_legend) guide_legend(nrow=2,byrow=TRUE,override.aes=list(shape=16)) else 'none',
           shape=if(show_legend) guide_legend(nrow=1,override.aes=list(colour=ink)) else 'none') +
    labs(x='Coverage (%)',y=NULL,title=title) + theme_pub +
    coord_cartesian(xlim=range(c(d$value-d$sd,d$value+d$sd,90))+c(-2,2))
}
a <- coverage_panel('coverage','a  Marginal coverage',TRUE)
b <- coverage_panel('block_coverage_min','b  Minimum tile coverage')
trade <- merge(agg('coverage'),agg('mean_set_size'),by=c('Predictor','Rule'),suffixes=c('_coverage','_size'))
trade <- subset(trade, Rule=='BC-MPS' | (Predictor=='Fused' & Rule=='Global'))
c <- ggplot(trade,aes(value_size,100*value_coverage,colour=Predictor,shape=Rule)) +
  geom_hline(yintercept=90,linetype='dashed',colour='#75818A',linewidth=.4) +
  geom_errorbar(aes(xmin=value_size-sd_size,xmax=value_size+sd_size),orientation='y',width=1,linewidth=.45) +
  geom_errorbar(aes(ymin=100*(value_coverage-sd_coverage),ymax=100*(value_coverage+sd_coverage)),width=.08,linewidth=.45) +
  geom_point(size=2.4) + scale_colour_manual(values=palette,breaks=names(palette),drop=FALSE) +
  scale_shape_manual(values=c(Global=1,'BC-MPS'=16)) + guides(colour='none',shape='none') +
  labs(x='Mean prediction-set size',y='Coverage (%)',title='c  Coverage versus set size') + theme_pub +
  coord_cartesian(ylim=range(100*c(trade$value_coverage-trade$sd_coverage,trade$value_coverage+trade$sd_coverage),90)+c(-2,2))
fig1 <- (a|b|c) + plot_layout(widths=c(1,1,1.2),guides='collect')
fig1 <- fig1 & theme(legend.position='bottom')

methods <- c('pointwise_global','bc_mps','sacp_global','sacp_tile_diagnostic')
names_m <- c('Pointwise global','BC-MPS','Adapted SACP global','SACP + tile diagnostic')
grid$Method <- factor(grid$method, levels=methods, labels=names_m)
grid$nominal <- 100*(1-grid$alpha); grid$empirical <- 100*grid$coverage
colors <- setNames(c('#75818A','#168A9A','#D06E6B','#647DB3'),names_m)
primary <- subset(grid,alpha==.1)
d <- ggplot(grid,aes(nominal,empirical,colour=Method)) +
  geom_abline(slope=1,intercept=0,colour='#AEB8C0',linewidth=.4) +
  geom_line(linewidth=.6) + geom_point(size=1.5) +
  geom_point(data=primary,shape=22,fill='white',size=2.8) + scale_colour_manual(values=colors) +
  labs(x='Nominal coverage (%)',y='Empirical coverage (%)',title='a  Nominal versus empirical coverage') + theme_pub
e <- ggplot(grid,aes(empirical,mean_set_size,colour=Method)) +
  geom_path(linewidth=.6) + geom_point(size=1.5) +
  geom_point(data=primary,shape=22,fill='white',size=2.8) + scale_colour_manual(values=colors) +
  labs(x='Empirical coverage (%)',y='Mean prediction-set size',title='b  Coverage and set-size trade-off') + theme_pub
fig2 <- (d|e) + plot_layout(guides='collect')
fig2 <- fig2 & theme(legend.position='bottom') & guides(colour=guide_legend(nrow=2,byrow=TRUE))
export <- function(stem,p,height) {
  ggsave(paste0(stem,'.pdf'),p,device=cairo_pdf,width=183,height=height,units='mm')
  ggsave(paste0(stem,'.svg'),p,device=svglite,width=183,height=height,units='mm')
  ggsave(paste0(stem,'.png'),p,device=agg_png,width=183,height=height,units='mm',dpi=300)
}
export('reproduced_figures/block_conformal_fixed30',fig1,104)
export('reproduced_figures/coverage_efficiency_fixed30',fig2,105)
write.csv(dat,'reproduced_figures/figure_source_primary.csv',row.names=FALSE)
write.csv(grid,'reproduced_figures/figure_source_grid.csv',row.names=FALSE)
cat('Exported both figures from fixed30 data; axes include all error bars.\n')
