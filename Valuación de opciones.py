#Modelar opciones en R Studio
install.packages("quantmod")
install.packages("TTR")
library(quantmod)
library(TTR)
#Obtener datos históricos de una acción (Apple Inc.)
getSymbols("AAPL", src= "yahoo", from="2020-01-02")
#Calcular la media móvil simple (SMA)
head(AAPL)
print(AAPL)
#Graficar datos históricos
chartSeries(AAPL)
sma<-SMA(Cl(AAPL), n = 20)
install.packages("RQuantLib")
library(RQuantLib)
#Especificar los parámetros de la opción
spot <-231.41 #Precio actual del activo
strike<-200.00 #Precio de ejercicio
maturity<- as.Date("2024-12-31")
volatility<-0.2
r<- 0.11 #Tasa Libre de Riesgo
#Calcular el precio de la opción de compra (call) usando Black-Scholes
option_price<-EuropeanOption(type="call", underlying = spot, strike=strike, dividendYield = 0, riskFreeRate = r, maturity = as.numeric(maturity - Sys.Date())/365,volatility = volatility) 
print(option_price)
