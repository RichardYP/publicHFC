# Indices enhancement strategy for benchmarking CSI 300 and CSI 500

Type: stock picking strategy

Design:



1. Obtain the cumulative net worth data of all hybrid and stock funds
  
  Data content:
  
    a) The accumulated net value on the day when the position is disclosed (2020-9-30, 2020-6-30, 2020-3-31, 2019-12-31)
    b) The cumulative net value of the position one year before the disclosure date (if it was established less than one year, 
    it will be recorded as the initial net value 1)
    c) Calculate the annual growth rate
    


2. Obtain quarterly disclosed stock position data
  
  Data content:
  
    a) Stock code, name, number of positions, market value of positions, 
    and percentage of the total market value of the fund
    b) Obtain the market value of all stocks of the fund in the quarter
    c) Calculate the proportion of stocks to the total stock market value



3. Determine the stock weight information of the benchmark CSI 300, CSI 500 or CSI 800 according to the fund performance comparison benchmark
  
          *This strategy only considers active funds, excluding theme funds, index tracking funds, quantitative funds, etc.,
          and only retains public funds that benchmark the above three indexes, and excludes other funds.



4. Compare the subject

      *Share%InStock: The stock is in the fund, accounting for% of the market value of all stocks

      *Share%InIndex: The stock is listed in CSI 300, CSI 500, or CSI 800, and accounts for the weighting% of the index

      *Share_Surplus: The overweight ratio of the stock compared to the index in the fund

          Share_Surplus = Share%InStock-Share%InIndex



5. Generation factor:

  Alpha_Size: The over-allocation value is weighted according to the market value of all stocks in the period held by the fund to obtain Whole_Weighted_Surplus
  
  Alpha_Median: Take the median of the overmatch value to get Median_Surplus
  
  Alpha_Rank: According to the fund's performance in the past year, the higher the performance ranking of the fund, the higher the weight of the overweight (still being processed)
  
  
            The final output factor is as follows:
            TradingDay SecuCode Alpha
            YYYY-MM-DD XXXXXX 0.XXX
                ... ... ...
      


6. Build a portfolio

       a) Arrange the stocks of different disclosure periods in descending order of Alpha value and divide them into 5 groups,
       each of which contains 20% of the stocks

       b) The number of stocks is allocated according to the weight of the stock’s Alpha in the total Alpha value, with a unit of 100

       c) Construct 25% CSI 300 ETF + 25% CSI 500 ETF + 50% overweight stock portfolio
  


7. Backtest

  Compare the portfolio yield and 50% CSI 300 ETF + 50% CSI 500 ETF yield to verify the effectiveness of overweight enhancement
