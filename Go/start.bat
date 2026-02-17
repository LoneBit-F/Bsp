@echo off
go build -o bot.exe bot.go
bot.exe %*
