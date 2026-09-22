python -m src.ps_dare.arima_pipeline --state-only-updates --jobs 4 /Users/tommasobertola/Git/ps-dare-paper-2/protocol-2-lag0.csv
keybase chat send tommasobertola 'done lag0 local pollution'

python -m src.ps_dare.arima_pipeline --state-only-updates --jobs 4 /Users/tommasobertola/Git/ps-dare-paper-2/protocol-2-lag1.csv
keybase chat send tommasobertola 'done lag1 local pollution'

python -m src.ps_dare.arima_pipeline --state-only-updates --jobs 4 /Users/tommasobertola/Git/ps-dare-paper-2/protocol-2-lag01.csv
keybase chat send tommasobertola 'done lag01 local pollution'
