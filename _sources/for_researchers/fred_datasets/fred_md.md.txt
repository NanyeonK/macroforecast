# FRED-MD

- Parent: [FRED datasets](index.md)
- Current dataset: FRED-MD

FRED-MD is the monthly national macro panel used by `dataset=fred_md`.
macroforecast downloads the official current CSV from:

`https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv`

Generated: `2026-04-30`. Current data through: `2025-09`. Current column
count excluding the date index: `126`.

## Column Contract

- Date column: `sasdate`, parsed as the monthly date index.
- Data columns: one column per FRED-MD mnemonic in the official current CSV.
- Transform row: `Transform:` gives the official FRED-MD T-code for each data
  column.
- Description source: official FRED-MD appendix when the mnemonic is present;
  otherwise the FRED series page or the official appendix/change log.

## All Current Columns

| # | Column | T-code | Transform | Group | Definition | Source |
|---:|---|---:|---|---|---|---|
| 1 | `RPI` | 5 | First log difference | Output and income | Real Personal Income | [FRED](https://fred.stlouisfed.org/series/RPI) |
| 2 | `W875RX1` | 5 | First log difference | Output and income | Real personal income ex transfer receipts | [FRED](https://fred.stlouisfed.org/series/W875RX1) |
| 3 | `DPCERA3M086SBEA` | 5 | First log difference | Consumption, orders, and inventories | Real personal consumption expenditures | [FRED](https://fred.stlouisfed.org/series/DPCERA3M086SBEA) |
| 4 | `CMRMTSPLx` | 5 | First log difference | Consumption, orders, and inventories | Real Manu. and Trade Industries Sales | constructed / appendix |
| 5 | `RETAILx` | 5 | First log difference | Consumption, orders, and inventories | Retail and Food Services Sales | constructed / appendix |
| 6 | `INDPRO` | 5 | First log difference | Output and income | IP Index | [FRED](https://fred.stlouisfed.org/series/INDPRO) |
| 7 | `IPFPNSS` | 5 | First log difference | Output and income | IP: Final Products and Nonindustrial Supplies | [FRED](https://fred.stlouisfed.org/series/IPFPNSS) |
| 8 | `IPFINAL` | 5 | First log difference | Output and income | IP: Final Products (Market Group) | [FRED](https://fred.stlouisfed.org/series/IPFINAL) |
| 9 | `IPCONGD` | 5 | First log difference | Output and income | IP: Consumer Goods | [FRED](https://fred.stlouisfed.org/series/IPCONGD) |
| 10 | `IPDCONGD` | 5 | First log difference | Output and income | IP: Durable Consumer Goods | [FRED](https://fred.stlouisfed.org/series/IPDCONGD) |
| 11 | `IPNCONGD` | 5 | First log difference | Output and income | IP: Nondurable Consumer Goods | [FRED](https://fred.stlouisfed.org/series/IPNCONGD) |
| 12 | `IPBUSEQ` | 5 | First log difference | Output and income | IP: Business Equipment | [FRED](https://fred.stlouisfed.org/series/IPBUSEQ) |
| 13 | `IPMAT` | 5 | First log difference | Output and income | IP: Materials | [FRED](https://fred.stlouisfed.org/series/IPMAT) |
| 14 | `IPDMAT` | 5 | First log difference | Output and income | IP: Durable Materials | [FRED](https://fred.stlouisfed.org/series/IPDMAT) |
| 15 | `IPNMAT` | 5 | First log difference | Output and income | IP: Nondurable Materials | [FRED](https://fred.stlouisfed.org/series/IPNMAT) |
| 16 | `IPMANSICS` | 5 | First log difference | Output and income | IP: Manufacturing (SIC) | [FRED](https://fred.stlouisfed.org/series/IPMANSICS) |
| 17 | `IPB51222S` | 5 | First log difference | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/IPB51222S) |
| 18 | `IPFUELS` | 5 | First log difference | Output and income | IP: Fuels | [FRED](https://fred.stlouisfed.org/series/IPFUELS) |
| 19 | `CUMFNS` | 2 | First difference | Output and income | Capacity Utilization: Manufacturing | [FRED](https://fred.stlouisfed.org/series/CUMFNS) |
| 20 | `HWI` | 2 | First difference | Labor market | Help-Wanted Index for United States | [FRED](https://fred.stlouisfed.org/series/HWI) |
| 21 | `HWIURATIO` | 2 | First difference | Labor market | Ratio of Help Wanted/No. Unemployed | [FRED](https://fred.stlouisfed.org/series/HWIURATIO) |
| 22 | `CLF16OV` | 5 | First log difference | Labor market | Civilian Labor Force | [FRED](https://fred.stlouisfed.org/series/CLF16OV) |
| 23 | `CE16OV` | 5 | First log difference | Labor market | Civilian Employment | [FRED](https://fred.stlouisfed.org/series/CE16OV) |
| 24 | `UNRATE` | 2 | First difference | Labor market | Civilian Unemployment Rate | [FRED](https://fred.stlouisfed.org/series/UNRATE) |
| 25 | `UEMPMEAN` | 2 | First difference | Labor market | Average Duration of Unemployment (Weeks) | [FRED](https://fred.stlouisfed.org/series/UEMPMEAN) |
| 26 | `UEMPLT5` | 5 | First log difference | Labor market | Civilians Unemployed - Less Than 5 Weeks | [FRED](https://fred.stlouisfed.org/series/UEMPLT5) |
| 27 | `UEMP5TO14` | 5 | First log difference | Labor market | Civilians Unemployed for 5-14 Weeks | [FRED](https://fred.stlouisfed.org/series/UEMP5TO14) |
| 28 | `UEMP15OV` | 5 | First log difference | Labor market | Civilians Unemployed - 15 Weeks & Over | [FRED](https://fred.stlouisfed.org/series/UEMP15OV) |
| 29 | `UEMP15T26` | 5 | First log difference | Labor market | Civilians Unemployed for 15-26 Weeks | [FRED](https://fred.stlouisfed.org/series/UEMP15T26) |
| 30 | `UEMP27OV` | 5 | First log difference | Labor market | Civilians Unemployed for 27 Weeks and Over | [FRED](https://fred.stlouisfed.org/series/UEMP27OV) |
| 31 | `CLAIMSx` | 5 | First log difference | Labor market | Initial Claims | constructed / appendix |
| 32 | `PAYEMS` | 5 | First log difference | Labor market | All Employees: Total nonfarm | [FRED](https://fred.stlouisfed.org/series/PAYEMS) |
| 33 | `USGOOD` | 5 | First log difference | Labor market | All Employees: Goods-Producing Industries | [FRED](https://fred.stlouisfed.org/series/USGOOD) |
| 34 | `CES1021000001` | 5 | First log difference | Labor market | All Employees: Mining and Logging: Mining | [FRED](https://fred.stlouisfed.org/series/CES1021000001) |
| 35 | `USCONS` | 5 | First log difference | Labor market | All Employees: Construction | [FRED](https://fred.stlouisfed.org/series/USCONS) |
| 36 | `MANEMP` | 5 | First log difference | Labor market | All Employees: Manufacturing | [FRED](https://fred.stlouisfed.org/series/MANEMP) |
| 37 | `DMANEMP` | 5 | First log difference | Labor market | All Employees: Durable goods | [FRED](https://fred.stlouisfed.org/series/DMANEMP) |
| 38 | `NDMANEMP` | 5 | First log difference | Labor market | All Employees: Nondurable goods | [FRED](https://fred.stlouisfed.org/series/NDMANEMP) |
| 39 | `SRVPRD` | 5 | First log difference | Labor market | All Employees: Service-Providing Industries | [FRED](https://fred.stlouisfed.org/series/SRVPRD) |
| 40 | `USTPU` | 5 | First log difference | Labor market | All Employees: Trade, Transportation & Utilities | [FRED](https://fred.stlouisfed.org/series/USTPU) |
| 41 | `USWTRADE` | 5 | First log difference | Labor market | All Employees: Wholesale Trade | [FRED](https://fred.stlouisfed.org/series/USWTRADE) |
| 42 | `USTRADE` | 5 | First log difference | Labor market | All Employees: Retail Trade | [FRED](https://fred.stlouisfed.org/series/USTRADE) |
| 43 | `USFIRE` | 5 | First log difference | Labor market | All Employees: Financial Activities | [FRED](https://fred.stlouisfed.org/series/USFIRE) |
| 44 | `USGOVT` | 5 | First log difference | Labor market | All Employees: Government | [FRED](https://fred.stlouisfed.org/series/USGOVT) |
| 45 | `CES0600000007` | 1 | No transformation | Labor market | Avg Weekly Hours : Goods-Producing | [FRED](https://fred.stlouisfed.org/series/CES0600000007) |
| 46 | `AWOTMAN` | 2 | First difference | Labor market | Avg Weekly Overtime Hours : Manufacturing | [FRED](https://fred.stlouisfed.org/series/AWOTMAN) |
| 47 | `AWHMAN` | 1 | No transformation | Labor market | Avg Weekly Hours : Manufacturing | [FRED](https://fred.stlouisfed.org/series/AWHMAN) |
| 48 | `HOUST` | 4 | Log level | Housing | Housing Starts: Total New Privately Owned | [FRED](https://fred.stlouisfed.org/series/HOUST) |
| 49 | `HOUSTNE` | 4 | Log level | Housing | Housing Starts, Northeast | [FRED](https://fred.stlouisfed.org/series/HOUSTNE) |
| 50 | `HOUSTMW` | 4 | Log level | Housing | Housing Starts, Midwest | [FRED](https://fred.stlouisfed.org/series/HOUSTMW) |
| 51 | `HOUSTS` | 4 | Log level | Housing | Housing Starts, South | [FRED](https://fred.stlouisfed.org/series/HOUSTS) |
| 52 | `HOUSTW` | 4 | Log level | Housing | Housing Starts, West | [FRED](https://fred.stlouisfed.org/series/HOUSTW) |
| 53 | `PERMIT` | 4 | Log level | Housing | New Private Housing Permits (SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMIT) |
| 54 | `PERMITNE` | 4 | Log level | Housing | New Private Housing Permits, Northeast (SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITNE) |
| 55 | `PERMITMW` | 4 | Log level | Housing | New Private Housing Permits, Midwest (SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITMW) |
| 56 | `PERMITS` | 4 | Log level | Housing | New Private Housing Permits, South (SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITS) |
| 57 | `PERMITW` | 4 | Log level | Housing | New Private Housing Permits, West (SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITW) |
| 58 | `ACOGNO` | 5 | First log difference | Consumption, orders, and inventories | New Orders for Consumer Goods | [FRED](https://fred.stlouisfed.org/series/ACOGNO) |
| 59 | `AMDMNOx` | 5 | First log difference | Consumption, orders, and inventories | New Orders for Durable Goods | constructed / appendix |
| 60 | `ANDENOx` | 5 | First log difference | Consumption, orders, and inventories | New Orders for Nondefense Capital Goods | constructed / appendix |
| 61 | `AMDMUOx` | 5 | First log difference | Consumption, orders, and inventories | Unfilled Orders for Durable Goods | constructed / appendix |
| 62 | `BUSINVx` | 5 | First log difference | Consumption, orders, and inventories | Total Business Inventories | constructed / appendix |
| 63 | `ISRATIOx` | 2 | First difference | Consumption, orders, and inventories | Total Business: Inventories to Sales Ratio | constructed / appendix |
| 64 | `M1SL` | 6 | Second log difference | Money and credit | M1 Money Stock | [FRED](https://fred.stlouisfed.org/series/M1SL) |
| 65 | `M2SL` | 6 | Second log difference | Money and credit | M2 Money Stock | [FRED](https://fred.stlouisfed.org/series/M2SL) |
| 66 | `M2REAL` | 5 | First log difference | Money and credit | Real M2 Money Stock | [FRED](https://fred.stlouisfed.org/series/M2REAL) |
| 67 | `BOGMBASE` | 6 | Second log difference | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/BOGMBASE) |
| 68 | `TOTRESNS` | 6 | Second log difference | Money and credit | Total Reserves of Depository Institutions | [FRED](https://fred.stlouisfed.org/series/TOTRESNS) |
| 69 | `NONBORRES` | 7 | First difference of percent change | Money and credit | Reserves Of Depository Institutions | [FRED](https://fred.stlouisfed.org/series/NONBORRES) |
| 70 | `BUSLOANS` | 6 | Second log difference | Money and credit | DC&I loans | [FRED](https://fred.stlouisfed.org/series/BUSLOANS) |
| 71 | `REALLN` | 6 | Second log difference | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/REALLN) |
| 72 | `NONREVSL` | 6 | Second log difference | Money and credit | Total Nonrevolving Credit | [FRED](https://fred.stlouisfed.org/series/NONREVSL) |
| 73 | `CONSPI` | 2 | First difference | Money and credit | Nonrevolving consumer credit to Personal Income | [FRED](https://fred.stlouisfed.org/series/CONSPI) |
| 74 | `S&P 500` | 5 | First log difference | Stock market | S&P’s Common Stock Price Index: Composite | constructed / appendix |
| 75 | `S&P div yield` | 2 | First difference | Stock market | S&P’s Composite Common Stock: Dividend Yield | constructed / appendix |
| 76 | `S&P PE ratio` | 5 | First log difference | Stock market | S&P’s Composite Common Stock: Price-Earnings Ratio | constructed / appendix |
| 77 | `FEDFUNDS` | 2 | First difference | Interest and exchange rates | Baa-FF spread | [FRED](https://fred.stlouisfed.org/series/FEDFUNDS) |
| 78 | `CP3Mx` | 2 | First difference | Interest and exchange rates | 3-Month AA Financial Commercial Paper Rate | constructed / appendix |
| 79 | `TB3MS` | 2 | First difference | Interest and exchange rates | 3-Month Treasury Bill: | [FRED](https://fred.stlouisfed.org/series/TB3MS) |
| 80 | `TB6MS` | 2 | First difference | Interest and exchange rates | 6-Month Treasury Bill: | [FRED](https://fred.stlouisfed.org/series/TB6MS) |
| 81 | `GS1` | 2 | First difference | Interest and exchange rates | 1-Year Treasury Rate | [FRED](https://fred.stlouisfed.org/series/GS1) |
| 82 | `GS5` | 2 | First difference | Interest and exchange rates | 5-Year Treasury Rate | [FRED](https://fred.stlouisfed.org/series/GS5) |
| 83 | `GS10` | 2 | First difference | Interest and exchange rates | 10-Year Treasury Rate | [FRED](https://fred.stlouisfed.org/series/GS10) |
| 84 | `AAA` | 2 | First difference | Interest and exchange rates | Moody’s Seasoned Aaa Corporate Bond Yield | [FRED](https://fred.stlouisfed.org/series/AAA) |
| 85 | `BAA` | 2 | First difference | Interest and exchange rates | Moody’s Seasoned Baa Corporate Bond Yield | [FRED](https://fred.stlouisfed.org/series/BAA) |
| 86 | `COMPAPFFx` | 1 | No transformation | Interest and exchange rates | 3-Month Commercial Paper Minus FEDFUNDS | constructed / appendix |
| 87 | `TB3SMFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/TB3SMFFM) |
| 88 | `TB6SMFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/TB6SMFFM) |
| 89 | `T1YFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/T1YFFM) |
| 90 | `T5YFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/T5YFFM) |
| 91 | `T10YFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/T10YFFM) |
| 92 | `AAAFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/AAAFFM) |
| 93 | `BAAFFM` | 1 | No transformation | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/BAAFFM) |
| 94 | `TWEXAFEGSMTHx` | 5 | First log difference | - | See official appendix / FRED source page. | constructed / appendix |
| 95 | `EXSZUSx` | 5 | First log difference | Interest and exchange rates | Switzerland / U.S. Foreign Exchange Rate | constructed / appendix |
| 96 | `EXJPUSx` | 5 | First log difference | Interest and exchange rates | Japan / U.S. Foreign Exchange Rate | constructed / appendix |
| 97 | `EXUSUKx` | 5 | First log difference | Interest and exchange rates | U.S. / U.K. Foreign Exchange Rate | constructed / appendix |
| 98 | `EXCAUSx` | 5 | First log difference | Interest and exchange rates | Canada / U.S. Foreign Exchange Rate | constructed / appendix |
| 99 | `WPSFD49207` | 6 | Second log difference | Prices | PPI: Finished Goods | [FRED](https://fred.stlouisfed.org/series/WPSFD49207) |
| 100 | `WPSFD49502` | 6 | Second log difference | Prices | PPI: Finished Consumer Goods | [FRED](https://fred.stlouisfed.org/series/WPSFD49502) |
| 101 | `WPSID61` | 6 | Second log difference | Prices | PPI: Intermediate Materials | [FRED](https://fred.stlouisfed.org/series/WPSID61) |
| 102 | `WPSID62` | 6 | Second log difference | Prices | PPI: Crude Materials | [FRED](https://fred.stlouisfed.org/series/WPSID62) |
| 103 | `OILPRICEx` | 6 | Second log difference | Prices | Crude Oil, spliced WTI and Cushing | constructed / appendix |
| 104 | `PPICMM` | 6 | Second log difference | Prices | PPI: Metals and metal products: | [FRED](https://fred.stlouisfed.org/series/PPICMM) |
| 105 | `CPIAUCSL` | 6 | Second log difference | Prices | CPI : All Items | [FRED](https://fred.stlouisfed.org/series/CPIAUCSL) |
| 106 | `CPIAPPSL` | 6 | Second log difference | Prices | CPI : Apparel | [FRED](https://fred.stlouisfed.org/series/CPIAPPSL) |
| 107 | `CPITRNSL` | 6 | Second log difference | Prices | CPI : Transportation | [FRED](https://fred.stlouisfed.org/series/CPITRNSL) |
| 108 | `CPIMEDSL` | 6 | Second log difference | Prices | CPI : Medical Care | [FRED](https://fred.stlouisfed.org/series/CPIMEDSL) |
| 109 | `CUSR0000SAC` | 6 | Second log difference | Prices | CPI : Commodities | [FRED](https://fred.stlouisfed.org/series/CUSR0000SAC) |
| 110 | `CUSR0000SAD` | 6 | Second log difference | Prices | CPI : Durables | [FRED](https://fred.stlouisfed.org/series/CUSR0000SAD) |
| 111 | `CUSR0000SAS` | 6 | Second log difference | Prices | CPI : Services | [FRED](https://fred.stlouisfed.org/series/CUSR0000SAS) |
| 112 | `CPIULFSL` | 6 | Second log difference | Prices | CPI : All Items Less Food | [FRED](https://fred.stlouisfed.org/series/CPIULFSL) |
| 113 | `CUSR0000SA0L2` | 6 | Second log difference | Prices | CPI : All items less shelter | [FRED](https://fred.stlouisfed.org/series/CUSR0000SA0L2) |
| 114 | `CUSR0000SA0L5` | 6 | Second log difference | Prices | CPI : All items less medical care | [FRED](https://fred.stlouisfed.org/series/CUSR0000SA0L5) |
| 115 | `PCEPI` | 6 | Second log difference | Prices | Personal Cons. Expend.: Chain Index | [FRED](https://fred.stlouisfed.org/series/PCEPI) |
| 116 | `DDURRG3M086SBEA` | 6 | Second log difference | Prices | Personal Cons. Exp: Durable goods | [FRED](https://fred.stlouisfed.org/series/DDURRG3M086SBEA) |
| 117 | `DNDGRG3M086SBEA` | 6 | Second log difference | Prices | Personal Cons. Exp: Nondurable goods | [FRED](https://fred.stlouisfed.org/series/DNDGRG3M086SBEA) |
| 118 | `DSERRG3M086SBEA` | 6 | Second log difference | Prices | Personal Cons. Exp: Services | [FRED](https://fred.stlouisfed.org/series/DSERRG3M086SBEA) |
| 119 | `CES0600000008` | 6 | Second log difference | Labor market | Avg Hourly Earnings : Goods-Producing | [FRED](https://fred.stlouisfed.org/series/CES0600000008) |
| 120 | `CES2000000008` | 6 | Second log difference | Labor market | Avg Hourly Earnings : Construction | [FRED](https://fred.stlouisfed.org/series/CES2000000008) |
| 121 | `CES3000000008` | 6 | Second log difference | Labor market | Avg Hourly Earnings : Manufacturing | [FRED](https://fred.stlouisfed.org/series/CES3000000008) |
| 122 | `UMCSENTx` | 2 | First difference | Consumption, orders, and inventories | Consumer Sentiment Index | constructed / appendix |
| 123 | `DTCOLNVHFNM` | 6 | Second log difference | Money and credit | Consumer Motor Vehicle Loans Outstanding | [FRED](https://fred.stlouisfed.org/series/DTCOLNVHFNM) |
| 124 | `DTCTHFNM` | 6 | Second log difference | Money and credit | Total Consumer Loans and Leases Outstanding | [FRED](https://fred.stlouisfed.org/series/DTCTHFNM) |
| 125 | `INVEST` | 6 | Second log difference | Money and credit | Securities in Bank Credit at All Commercial Banks | [FRED](https://fred.stlouisfed.org/series/INVEST) |
| 126 | `VIXCLSx` | 1 | No transformation | - | See official appendix / FRED source page. | constructed / appendix |
