setwd("/Users/Yuliya/Dropbox/Research/VU/GPs/DoseGP_script")

data = read.table('cokol_full.csv', sep=',',header = TRUE)
Response = data$ABA.WOR.response
DoseA = data$ABA.WOR.doseRow
DoseB = data$ABA.WOR.doseCol

xgrid$x <- matrix(DoseA, c(8,8))
xgrid$y <- matrix(DoseB, c(8,8))
ygrid <- matrix(Response, c(8,8))

xdesign <- cbind(DoseA, DoseB)
ydesign <- Response

#### 2D constrained model ####
# creating the model
model <- create(class = 'lineqGP', x = xdesign, y = ydesign,
                constrType = c('monotonicity'))
model$localParam$m <- 10
model$bounds[1, ] <- c(lower = 0, upper = 2.0)
model$kernParam$par <- c(sigma2 = 1^2, theta = 10.2)
model <- augment(model)

# evaluating the model
ntest <- 10
xtest  <- as.matrix(expand.grid(seq(0, 280, length = ntest), seq(0, 280, length = ntest)))
ytest <- atan(5*xtest[,1]) + atan(xtest[,2])
pred <- predict(model, xtest)


#### Optimizing the model ####
model2 <- lineqGPOptim(model,
                       x0 = model$kernParam$par,
                       eval_f = "logLik",
                       add.constr = FALSE,
                       mcmc.opts = list(probe = "Genz", nb.mcmc = 1e3),
                       opts = list(algorithm = "NLOPT_LD_MMA",
                                   print_level = 3,
                                   ftol_abs = 1e-3,
                                   maxeval = 50,
                                   check_derivatives = TRUE,
                                   #estim.varnoise = TRUE,
                                   parfixed = c(FALSE, FALSE, FALSE)),
                       lb = rep(0.1, 3),
                       ub = c(2, 0.9, 0.9),
                       max.trials = 10)

# simulating samples from the optimal model
pred2 <- predict(model2, xtest)
sim.model2 <- simulate(model2, nsim = 1e1, seed = 7, xtest = xtest)
p <- persp3D(x = seq(0, 1, length = ntest), y = seq(0, 1, length = ntest),
             z = matrix(pred2$Phi.test %*% pred2$xi.map, nrow = ntest),
             xlab = "x1", ylab = "x2", zlab = "y(x1,x2)", zlim = c(-0.2,2.2),
             main = "Constrained GP with covariance parameter estimation",
             phi = 20, theta = -30, col = colormap, alpha = 1, colkey = FALSE,
             image = TRUE, contour = TRUE)
for (k in c(1,ncol(sim.model2$ysim))){
  persp3D(x = seq(0, 1, length = ntest),
          y = seq(0, 1, length = ntest),
          z = matrix(sim.model2$ysim[,k], nrow = ntest),
          phi = 20, theta = -30, col = colormap, alpha = 0.1,
          colkey = FALSE, add = TRUE)
}
image3D(x = seq(0, 1, length = 10), y = seq(0, 1, length = 10), z = model$bounds[1,1],
        phi = 20, theta = -30, col = "black", alpha = 0.1, add = TRUE)
image3D(x = seq(0, 1, length = 10), y = seq(0, 1, length = 10), z = model$bounds[1,2],
        phi = 20, theta = -30, col = "black", alpha = 0.1, add = TRUE)
points(trans3D(x = xdesign[,1], y = xdesign[,2], z = ydesign, pmat = p),
       col = 'black', pch = 19, cex = 1.0)