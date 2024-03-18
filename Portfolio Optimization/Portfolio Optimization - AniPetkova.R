#Checking if all necessary libraries are installed
if (FALSE==is.element("PortfolioAnalytics", installed.packages()[,1])) {
  install.packages("PortfolioAnalytics")
}
if (FALSE==is.element("quantmod", installed.packages()[,1])) {
  install.packages("quantmod")
}
if (FALSE==is.element("PerformanceAnalytics", installed.packages()[,1])) {
  install.packages("PerformanceAnalytics")
}
if (FALSE==is.element("zoo", installed.packages()[,1])) {
  install.packages("zoo")
}
if (FALSE==is.element("plotly", installed.packages()[,1])) {
  install.packages("plotly")
}
if (FALSE==is.element("DEoptim", installed.packages()[,1])) {
  install.packages("DEoptim")
}

library(PortfolioAnalytics)
library(quantmod)
library(PerformanceAnalytics)
library(zoo)
library(plotly)
library(DEoptim)
library(ggplot2)
library(magrittr)
library(broom)


### Downloading the data ###
AAPL <- getSymbols("AAPL", auto.assign = FALSE, from = '2020-03-01', to = '2023-03-01')
MNST <- getSymbols("MNST", auto.assign = FALSE, from = '2020-03-01', to = '2023-03-01')
NFLX <- getSymbols("NFLX", auto.assign = FALSE, from = '2020-03-01', to = '2023-03-01')
PYPL <- getSymbols("PYPL", auto.assign = FALSE, from = '2020-03-01', to = '2023-03-01')
AZN <- getSymbols("AZN", auto.assign = FALSE, from = '2020-03-01', to = '2023-03-01')

### Merging all data into a matrix ###
prices.data <- merge.zoo(AAPL[,6], MNST[,6], NFLX[,6], PYPL[,6], AZN[,6])

### Calculating returns ###
returns.data <- ROC(prices.data)
returns.data <- na.omit(returns.data)
plot(returns.data)
max(returns.data)

### Naming each column of the dataset ###
colnames(returns.data) <- c("AAPL", "MNST", "NFLX", "PYPL", "AZN")

### Looking for the maximum price of each company ###
max(prices.data[,1])
max(prices.data[,2])
max(prices.data[,3])
max(prices.data[,4])
max(prices.data[,5])

### Plotting the daily prices ###
series = tidy(prices.data) %>% 
  
  ggplot(aes(x=index,y=value, color=series)) + 
  geom_line() +
  facet_grid(series~.,scales = "free") + 
  labs(title = "Daily Prices Data",
       
       subtitle = "End of Day Adjusted Prices",
       caption = " Source: Yahoo Finance") +
  
  xlab("Date") + ylab("Price") +
  scale_color_manual(values = c("Red", "Black", "DarkBlue","Orange", "Green"))
series


### Plotting the daily returns ###
series2 = tidy(returns.data) %>% 
  
  ggplot(aes(x=index,y=value, color=series)) + 
  geom_line() +
  facet_grid(series~.,scales = "free") + 
  labs(title = "Daily Returns Data",
       
       subtitle = "End of Day Adjusted Prices",
       caption = " Source: Yahoo Finance") +
  
  xlab("Date") + ylab("Price") +
  scale_color_manual(values = c("Red", "Black", "DarkBlue","Orange", "Green"))
series2


### Mean, SD and Covariance matrix ###
meanReturns <- colMeans(returns.data)
show(meanReturns)

SDReturns <- sapply(returns.data, sd)
show(SDReturns)
pie(SDReturns)

w = c(0.20, 0.20, 0.20, 0.20, 0.20)
PortfolioSD <-StdDev(returns.data, weights = w)

covMat <- cov(returns.data)
show(covMat)

### Creating a portfolio ###
port <- portfolio.spec(assets = c("AAPL", "MNST", "NFLX", "PYPL", "AZN"))


### Expected shortfall ###
esData = ES(returns.data, p=0.95,  method="modified", portfolio_method="component")

total <- sum(esData$contribution)

pct <- paste(colnames(returns.data), 100*(esData$contribution / total), sep=":")

pie(esData$contribution, labels=pct)
show(esData)



###Adding constraints and objectives to the portfolio###

### Box ### 
port <- add.constraint(port, type = "box", min = 0.05, max = 0.8)

### Leverage ###
port <- add.constraint(portfolio = port, type = "full_investment")

### Random portfolios ###
rportfolios <- random_portfolios(port, permutations = 1000, rp_method = "sample")

### Minimum risk portfolio ###
minvar.port <- add.objective(port, type = "risk", name = "var")
minvar.opt <- optimize.portfolio(returns.data, minvar.port, optimize_method = "random", 
                                 rp = rportfolios)

### Maximum return portfolio ###
maxret.port <- add.objective(port, type = "return", name = "mean", val=0.0001)
maxret.opt <- optimize.portfolio(returns.data, maxret.port, optimize_method = "random", 
                
                                                 rp = rportfolios)
View(minvar.opt)
View(maxret.opt)

### 
minret <- sum(minvar.opt$weights * meanReturns) - 0.01
maxret <- sum(maxret.opt$weights * meanReturns) + 0.01

vec <- seq(minret, maxret, length.out = 20)

eff.frontier <- data.frame(Risk = rep(NA, length(vec)),
                           Return = rep(NA, length(vec)), 
                           SharpeRatio = rep(NA, length(vec)))

frontier.weights <- mat.or.vec(nr = length(vec), nc = ncol(returns.data))
colnames(frontier.weights) <- colnames(returns.data)

### Optimization ###
for(i in 1:length(vec)){
  eff.port <- add.constraint(port, type = "return", name = "mean", return_target = vec[i])
  eff.port <- add.objective(eff.port, type = "risk", name = "var")
  # eff.port <- add.objective(eff.port, type = "weight_concentration", name = "HHI",
  #                            conc_aversion = 0.001)
  
  eff.port <- optimize.portfolio(returns.data, eff.port)
  
  eff.frontier$Risk[i] <- sqrt(t(eff.port$weights) %*% covMat %*% eff.port$weights)
  
  eff.frontier$Return[i] <- eff.port$weights %*% meanReturns
  
  eff.frontier$Sharperatio[i] <- eff.port$Return[i] / eff.port$Risk[i]
  
  frontier.weights[i,] = eff.port$weights
  
  print(paste(round(i/length(vec) * 100, 0), "% done..."))
}

feasible.sd <- apply(rportfolios, 1, function(x){
  return(sqrt(matrix(x, nrow = 1) %*% covMat %*% matrix(x, ncol = 1)))
})

feasible.means <- apply(rportfolios, 1, function(x){
  return(x %*% meanReturns)
})

feasible.sr <- feasible.means / feasible.sd


p <- plot_ly(x = feasible.sd, y = feasible.means, 
             mode = "markers", type = "scattergl", showlegend = F,
             
             marker = list(size = 3, opacity = 0.7, 
                           colorbar = list(title = "Sharpe Ratio"))) %>% 
  
  add_trace(data = eff.frontier, x = ~Risk, y = ~Return, mode = "markers", 
            type = "scattergl", showlegend = F, 
            marker = list(color = "#000000", size = 5)) %>% 
  
  
  
  print(p)






