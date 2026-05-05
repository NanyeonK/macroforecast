# 5.2 FRED-QD

- Parent: [5. FRED-Dataset](index.md)
- Current dataset: FRED-QD

FRED-QD is the quarterly national macro panel used by `dataset=fred_qd`.
macroforecast downloads the official current CSV from:

`https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv`

Generated: `2026-04-30`. Current data through: `2025-09`. Current column
count excluding the date index: `245`.

## Column Contract

- Date column: `sasdate`, parsed as the quarterly date index.
- Data columns: one column per FRED-QD mnemonic in the official current CSV.
- `factors` row: `1` means the series is in the Stock-Watson-style factor
  construction set; `0` means it is not.
- Transform row: `transform` gives the official FRED-QD T-code for each data
  column.
- Description source: official FRED-QD appendix when the mnemonic is present;
  otherwise the FRED series page or the official appendix/change log.

## All Current Columns

| # | Column | T-code | Transform | SW factor | Group | Definition | Source |
|---:|---|---:|---|---:|---|---|---|
| 1 | `GDPC1` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/GDPC1) |
| 2 | `PCECC96` | 5 | First log difference | 0 | NIPA | Real Personal Consumption Expenditures (Billions of Chained 2009 Dollars) | [FRED](https://fred.stlouisfed.org/series/PCECC96) |
| 3 | `PCDGx` | 5 | First log difference | 1 | NIPA | Real personal consumption expenditures: Durable goods (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 4 | `PCESVx` | 5 | First log difference | 1 | NIPA | Real Personal Consumption Expenditures: Services (Billions of 2009 Dollars), deflated using PCE | constructed / appendix |
| 5 | `PCNDx` | 5 | First log difference | 1 | NIPA | Real Personal Consumption Expenditures: Nondurable Goods (Billions of 2009 Dollars), deflated using PCE | constructed / appendix |
| 6 | `GPDIC1` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/GPDIC1) |
| 7 | `FPIx` | 5 | First log difference | 0 | NIPA | Real private fixed investment (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 8 | `Y033RC1Q027SBEAx` | 5 | First log difference | 1 | NIPA | Inv:Equip&Software Real Gross Private Domestic Investment: Fixed Investment: Nonresidential: Equipment (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 9 | `PNFIx` | 5 | First log difference | 1 | NIPA | Real private fixed investment: Nonresidential (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 10 | `PRFIx` | 5 | First log difference | 1 | NIPA | Real private fixed investment: Residential (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 11 | `A014RE1Q156NBEA` | 1 | No transformation | 1 | NIPA | Shares of gross domestic product: Gross private domestic investment: Change in private inventories (Percent) | [FRED](https://fred.stlouisfed.org/series/A014RE1Q156NBEA) |
| 12 | `GCEC1` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/GCEC1) |
| 13 | `A823RL1Q225SBEA` | 1 | No transformation | 1 | NIPA | Real Government Consumption Expenditures and Gross Investment: Federal (Percent Change from Preceding Period) | [FRED](https://fred.stlouisfed.org/series/A823RL1Q225SBEA) |
| 14 | `FGRECPTx` | 5 | First log difference | 1 | NIPA | Real Federal Government Current Receipts (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 15 | `SLCEx` | 5 | First log difference | 1 | NIPA | Real government state and local consumption expenditures (Billions of Chained 2009 Dollars), deflated using PCE | constructed / appendix |
| 16 | `EXPGSC1` | 5 | First log difference | 1 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/EXPGSC1) |
| 17 | `IMPGSC1` | 5 | First log difference | 1 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/IMPGSC1) |
| 18 | `DPIC96` | 5 | First log difference | 0 | NIPA | Real Disposable Personal Income (Billions of Chained 2009 Dollars) | [FRED](https://fred.stlouisfed.org/series/DPIC96) |
| 19 | `OUTNFB` | 5 | First log difference | 0 | NIPA | Nonfarm Business Sector: Real Output (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/OUTNFB) |
| 20 | `OUTBS` | 5 | First log difference | 0 | NIPA | Business Sector: Real Output (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/OUTBS) |
| 21 | `OUTMS` | 5 | First log difference | 0 | NIPA | Manufacturing Sector: Real Output (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/OUTMS) |
| 22 | `INDPRO` | 5 | First log difference | 0 | Industrial Production | Industrial Production Index (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/INDPRO) |
| 23 | `IPFINAL` | 5 | First log difference | 0 | Industrial Production | Industrial Production: Final Products (Market Group) (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPFINAL) |
| 24 | `IPCONGD` | 5 | First log difference | 0 | Industrial Production | Industrial Production: Consumer Goods (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPCONGD) |
| 25 | `IPMAT` | 5 | First log difference | 0 | Industrial Production | Industrial Production: Materials (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPMAT) |
| 26 | `IPDMAT` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Durable Materials (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPDMAT) |
| 27 | `IPNMAT` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Nondurable Materials (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPNMAT) |
| 28 | `IPDCONGD` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Durable Consumer Goods (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPDCONGD) |
| 29 | `IPB51110SQ` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Durable Goods: Automotive products (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPB51110SQ) |
| 30 | `IPNCONGD` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Nondurable Consumer Goods (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPNCONGD) |
| 31 | `IPBUSEQ` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Business Equipment (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPBUSEQ) |
| 32 | `IPB51220SQ` | 5 | First log difference | 1 | Industrial Production | Industrial Production: Consumer energy products (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPB51220SQ) |
| 33 | `TCU` | 1 | No transformation | 1 | Industrial Production | Capacity Utilization: Total Industry (Percent of Capacity) | [FRED](https://fred.stlouisfed.org/series/TCU) |
| 34 | `CUMFNS` | 1 | No transformation | 1 | Industrial Production | Capacity Utilization: Manufacturing (SIC) (Percent of Capacity) | [FRED](https://fred.stlouisfed.org/series/CUMFNS) |
| 35 | `PAYEMS` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Total nonfarm (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/PAYEMS) |
| 36 | `USPRIV` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Total Private Industries (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USPRIV) |
| 37 | `MANEMP` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Manufacturing (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/MANEMP) |
| 38 | `SRVPRD` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Service-Providing Industries (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/SRVPRD) |
| 39 | `USGOOD` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Goods-Producing Industries (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USGOOD) |
| 40 | `DMANEMP` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Durable goods (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/DMANEMP) |
| 41 | `NDMANEMP` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Nondurable goods (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/NDMANEMP) |
| 42 | `USCONS` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Construction (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USCONS) |
| 43 | `USEHS` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Education & Health Services (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USEHS) |
| 44 | `USFIRE` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Financial Activities (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USFIRE) |
| 45 | `USINFO` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Information Services (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USINFO) |
| 46 | `USPBS` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Professional & Business Services (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USPBS) |
| 47 | `USLAH` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Leisure & Hospitality (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USLAH) |
| 48 | `USSERV` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Other Services (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USSERV) |
| 49 | `USMINE` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Mining and logging (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USMINE) |
| 50 | `USTPU` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Trade, Transportation & Utilities (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USTPU) |
| 51 | `USGOVT` | 5 | First log difference | 0 | Employment and Unemployment | All Employees: Government (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USGOVT) |
| 52 | `USTRADE` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Retail Trade (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USTRADE) |
| 53 | `USWTRADE` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Wholesale Trade (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/USWTRADE) |
| 54 | `CES9091000001` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Government: Federal (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/CES9091000001) |
| 55 | `CES9092000001` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Government: State Government (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/CES9092000001) |
| 56 | `CES9093000001` | 5 | First log difference | 1 | Employment and Unemployment | All Employees: Government: Local Government (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/CES9093000001) |
| 57 | `CE16OV` | 5 | First log difference | 0 | Employment and Unemployment | Civilian Employment (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/CE16OV) |
| 58 | `CIVPART` | 2 | First difference | 0 | Employment and Unemployment | Civilian Labor Force Participation Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/CIVPART) |
| 59 | `UNRATE` | 2 | First difference | 0 | Employment and Unemployment | Civilian Unemployment Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/UNRATE) |
| 60 | `UNRATESTx` | 2 | First difference | 0 | Employment and Unemployment | Unemployment Rate less than 27 weeks (Percent) | constructed / appendix |
| 61 | `UNRATELTx` | 2 | First difference | 0 | Employment and Unemployment | Unemployment Rate for more than 27 weeks (Percent) | constructed / appendix |
| 62 | `LNS14000012` | 2 | First difference | 1 | Employment and Unemployment | Unemployment Rate - 16 to 19 years (Percent) | [FRED](https://fred.stlouisfed.org/series/LNS14000012) |
| 63 | `LNS14000025` | 2 | First difference | 1 | Employment and Unemployment | Unemployment Rate - 20 years and over, Men (Percent) | [FRED](https://fred.stlouisfed.org/series/LNS14000025) |
| 64 | `LNS14000026` | 2 | First difference | 1 | Employment and Unemployment | Unemployment Rate - 20 years and over, Women (Percent) | [FRED](https://fred.stlouisfed.org/series/LNS14000026) |
| 65 | `UEMPLT5` | 5 | First log difference | 1 | Employment and Unemployment | Number of Civilians Unemployed - Less Than 5 Weeks (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/UEMPLT5) |
| 66 | `UEMP5TO14` | 5 | First log difference | 1 | Employment and Unemployment | Number of Civilians Unemployed for 5 to 14 Weeks (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/UEMP5TO14) |
| 67 | `UEMP15T26` | 5 | First log difference | 1 | Employment and Unemployment | Number of Civilians Unemployed for 15 to 26 Weeks (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/UEMP15T26) |
| 68 | `UEMP27OV` | 5 | First log difference | 1 | Employment and Unemployment | Number of Civilians Unemployed for 27 Weeks and Over (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/UEMP27OV) |
| 69 | `LNS13023621` | 5 | First log difference | 1 | Employment and Unemployment | Unemployment Level - Job Losers (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/LNS13023621) |
| 70 | `LNS13023557` | 5 | First log difference | 1 | Employment and Unemployment | Unemployment Level - Reentrants to Labor Force (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/LNS13023557) |
| 71 | `LNS13023705` | 5 | First log difference | 1 | Employment and Unemployment | Unemployment Level - Job Leavers (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/LNS13023705) |
| 72 | `LNS13023569` | 5 | First log difference | 1 | Employment and Unemployment | Unemployment Level - New Entrants (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/LNS13023569) |
| 73 | `LNS12032194` | 5 | First log difference | 1 | Employment and Unemployment, continued | Employment Level - Part-Time for Economic Reasons, All Industries (Thousands of Persons) | [FRED](https://fred.stlouisfed.org/series/LNS12032194) |
| 74 | `HOABS` | 5 | First log difference | 0 | Employment and Unemployment, continued | EmpHrs:Bus Sec Business Sector: Hours of All Persons (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/HOABS) |
| 75 | `HOAMS` | 5 | First log difference | 0 | Employment and Unemployment, continued | Manufacturing Sector: Hours of All Persons (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/HOAMS) |
| 76 | `HOANBS` | 5 | First log difference | 0 | Employment and Unemployment, continued | Nonfarm Business Sector: Hours of All Persons (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/HOANBS) |
| 77 | `AWHMAN` | 1 | No transformation | 1 | Employment and Unemployment, continued | Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing (Hours) | [FRED](https://fred.stlouisfed.org/series/AWHMAN) |
| 78 | `AWHNONAG` | 2 | First difference | 1 | Employment and Unemployment, continued | Average Weekly Hours Of Production And Nonsupervisory Employees: Total private (Hours) | [FRED](https://fred.stlouisfed.org/series/AWHNONAG) |
| 79 | `AWOTMAN` | 2 | First difference | 1 | Employment and Unemployment, continued | AWH Overtime Average Weekly Overtime Hours of Production and Nonsupervisory Employees: Manufacturing (Hours) | [FRED](https://fred.stlouisfed.org/series/AWOTMAN) |
| 80 | `HWIx` | 1 | No transformation | 0 | Employment and Unemployment, continued | Help-Wanted Index | constructed / appendix |
| 81 | `HOUST` | 5 | First log difference | 0 | Housing | Housing Starts: Total: New Privately Owned Housing Units Started (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/HOUST) |
| 82 | `HOUST5F` | 5 | First log difference | 0 | Housing | Privately Owned Housing Starts: 5-Unit Structures or More (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/HOUST5F) |
| 83 | `PERMIT` | 5 | First log difference | 1 | Housing | New Private Housing Units Authorized by Building Permits (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/PERMIT) |
| 84 | `HOUSTMW` | 5 | First log difference | 1 | Housing | Housing Starts in Midwest Census Region (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/HOUSTMW) |
| 85 | `HOUSTNE` | 5 | First log difference | 1 | Housing | Housing Starts in Northeast Census Region (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/HOUSTNE) |
| 86 | `HOUSTS` | 5 | First log difference | 1 | Housing | Housing Starts in South Census Region (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/HOUSTS) |
| 87 | `HOUSTW` | 5 | First log difference | 1 | Housing | Housing Starts in West Census Region (Thousands of Units) | [FRED](https://fred.stlouisfed.org/series/HOUSTW) |
| 88 | `CMRMTSPLx` | 5 | First log difference | 0 | Inventories, Orders, and Sales | Real Manufacturing and Trade Industries Sales (Millions of Chained 2009 Dollars) | constructed / appendix |
| 89 | `RSAFSx` | 5 | First log difference | 1 | Inventories, Orders, and Sales | Real Retail and Food Services Sales (Millions of Chained 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 90 | `AMDMNOx` | 5 | First log difference | 1 | Inventories, Orders, and Sales | Real Manufacturers’ New Orders: Durable Goods (Millions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 91 | `ACOGNOx` | 5 | First log difference | 1 | Inventories, Orders, and Sales | Orders(ConsGoods/Mat.) Real Value of Manufacturers’ New Orders for Consumer Goods Industries (Million of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 92 | `AMDMUOx` | 5 | First log difference | 1 | Inventories, Orders, and Sales | Real Value of Manufacturers’ Unfilled Orders for Durable Goods Industries (Million of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 93 | `ANDENOx` | 5 | First log difference | 1 | Inventories, Orders, and Sales | Real Value of Manufacturers’ New Orders for Capital Goods: Nondefense Capital Goods Industries (Million of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 94 | `INVCQRMTSPL` | 5 | First log difference | 1 | Inventories, Orders, and Sales | Real Manufacturing and Trade Inventories (Millions of 2009 Dollars) | [FRED](https://fred.stlouisfed.org/series/INVCQRMTSPL) |
| 95 | `PCECTPI` | 6 | Second log difference | 0 | Prices | Personal Consumption Expenditures: Chain-type Price Index (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/PCECTPI) |
| 96 | `PCEPILFE` | 6 | Second log difference | 0 | Prices | Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index) (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/PCEPILFE) |
| 97 | `GDPCTPI` | 6 | Second log difference | 0 | Prices | Gross Domestic Product: Chain-type Price Index (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/GDPCTPI) |
| 98 | `GPDICTPI` | 6 | Second log difference | 1 | Prices | Gross Private Domestic Investment: Chain-type Price Index (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/GPDICTPI) |
| 99 | `IPDBS` | 6 | Second log difference | 1 | Prices | Business Sector: Implicit Price Deflator (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/IPDBS) |
| 100 | `DGDSRG3Q086SBEA` | 6 | Second log difference | 0 | Prices | Personal consumption expenditures: Goods (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DGDSRG3Q086SBEA) |
| 101 | `DDURRG3Q086SBEA` | 6 | Second log difference | 0 | Prices | Personal consumption expenditures: Durable goods (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DDURRG3Q086SBEA) |
| 102 | `DSERRG3Q086SBEA` | 6 | Second log difference | 0 | Prices | Personal consumption expenditures: Services (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DSERRG3Q086SBEA) |
| 103 | `DNDGRG3Q086SBEA` | 6 | Second log difference | 0 | Prices | Personal consumption expenditures: Nondurable goods (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DNDGRG3Q086SBEA) |
| 104 | `DHCERG3Q086SBEA` | 6 | Second log difference | 0 | Prices | Personal consumption expenditures: Services: Household consumption expenditures (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DHCERG3Q086SBEA) |
| 105 | `DMOTRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Durable goods: Motor vehicles and parts (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DMOTRG3Q086SBEA) |
| 106 | `DFDHRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Durable goods: Furnishings and durable household equipment (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DFDHRG3Q086SBEA) |
| 107 | `DREQRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Durable goods: Recreational goods and vehicles (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DREQRG3Q086SBEA) |
| 108 | `DODGRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Durable goods: Other durable goods (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DODGRG3Q086SBEA) |
| 109 | `DFXARG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Nondurable goods: Food and beverages purchased for off-premises consumption (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DFXARG3Q086SBEA) |
| 110 | `DCLORG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Nondurable goods: Clothing and footwear (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DCLORG3Q086SBEA) |
| 111 | `DGOERG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Nondurable goods: Gasoline and other energy goods (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DGOERG3Q086SBEA) |
| 112 | `DONGRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Nondurable goods: Other nondurable goods (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DONGRG3Q086SBEA) |
| 113 | `DHUTRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | PCED Housing-Utilities Personal consumption expenditures: Services: Housing and utilities (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DHUTRG3Q086SBEA) |
| 114 | `DHLCRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Services: Health care (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DHLCRG3Q086SBEA) |
| 115 | `DTRSRG3Q086SBEA` | 6 | Second log difference | 1 | Prices | Personal consumption expenditures: Transportation services (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DTRSRG3Q086SBEA) |
| 116 | `DRCARG3Q086SBEA` | 6 | Second log difference | 1 | Prices, continued | Personal consumption expenditures: Recreation services (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DRCARG3Q086SBEA) |
| 117 | `DFSARG3Q086SBEA` | 6 | Second log difference | 1 | Prices, continued | Personal consumption expenditures: Services: Food services and accommodations (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DFSARG3Q086SBEA) |
| 118 | `DIFSRG3Q086SBEA` | 6 | Second log difference | 1 | Prices, continued | Personal consumption expenditures: Financial services and insurance (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DIFSRG3Q086SBEA) |
| 119 | `DOTSRG3Q086SBEA` | 6 | Second log difference | 1 | Prices, continued | Personal consumption expenditures: Other services (chain-type price index) | [FRED](https://fred.stlouisfed.org/series/DOTSRG3Q086SBEA) |
| 120 | `CPIAUCSL` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: All Items (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CPIAUCSL) |
| 121 | `CPILFESL` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: All Items Less Food & Energy (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CPILFESL) |
| 122 | `WPSFD49207` | 6 | Second log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/WPSFD49207) |
| 123 | `PPIACO` | 6 | Second log difference | 0 | Prices, continued | Producer Price Index for All Commodities (Index 1982=100) | [FRED](https://fred.stlouisfed.org/series/PPIACO) |
| 124 | `WPSFD49502` | 6 | Second log difference | 1 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/WPSFD49502) |
| 125 | `WPSFD4111` | 6 | Second log difference | 1 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/WPSFD4111) |
| 126 | `PPIIDC` | 6 | Second log difference | 1 | Prices, continued | Producer Price Index by Commodity Industrial Commodities (Index 1982=100) | [FRED](https://fred.stlouisfed.org/series/PPIIDC) |
| 127 | `WPSID61` | 6 | Second log difference | 1 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/WPSID61) |
| 128 | `WPU0531` | 5 | First log difference | 1 | Prices, continued | Producer Price Index by Commodity for Fuels and Related Products and Power: Natural Gas (Index 1982=100) | [FRED](https://fred.stlouisfed.org/series/WPU0531) |
| 129 | `WPU0561` | 5 | First log difference | 1 | Prices, continued | Producer Price Index by Commodity for Fuels and Related Products and Power: Crude Petroleum (Domestic Production) (Index 1982=100) | [FRED](https://fred.stlouisfed.org/series/WPU0561) |
| 130 | `OILPRICEx` | 5 | First log difference | 0 | Prices, continued | Real Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma (2009 Dollars per Barrel), deflated by Core PCE | constructed / appendix |
| 131 | `AHETPIx` | 5 | First log difference | 0 | Earnings and Productivity | Real AHE:PrivInd Real Average Hourly Earnings of Production and Nonsupervisory Employees: Total Private (2009 Dollars per Hour), deflated by Core PCE | constructed / appendix |
| 132 | `CES2000000008x` | 5 | First log difference | 0 | Earnings and Productivity | Real Average Hourly Earnings of Production and Nonsupervisory Employees: Construction (2009 Dollars per Hour), deflated by Core PCE | constructed / appendix |
| 133 | `CES3000000008x` | 5 | First log difference | 0 | Earnings and Productivity | Real Average Hourly Earnings of Production and Nonsupervisory Employees: Manufacturing (2009 Dollars per Hour), deflated by Core PCE | constructed / appendix |
| 134 | `COMPRMS` | 5 | First log difference | 1 | Earnings and Productivity | Manufacturing Sector: Real Compensation Per Hour (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/COMPRMS) |
| 135 | `COMPRNFB` | 5 | First log difference | 1 | Earnings and Productivity | Nonfarm Business Sector: Real Compensation Per Hour (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/COMPRNFB) |
| 136 | `RCPHBS` | 5 | First log difference | 1 | Earnings and Productivity | Business Sector: Real Compensation Per Hour (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/RCPHBS) |
| 137 | `OPHMFG` | 5 | First log difference | 1 | Earnings and Productivity | Manufacturing Sector: Real Output Per Hour of All Persons (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/OPHMFG) |
| 138 | `OPHNFB` | 5 | First log difference | 1 | Earnings and Productivity | Nonfarm Business Sector: Real Output Per Hour of All Persons (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/OPHNFB) |
| 139 | `OPHPBS` | 5 | First log difference | 0 | Earnings and Productivity | Business Sector: Real Output Per Hour of All Persons (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/OPHPBS) |
| 140 | `ULCBS` | 5 | First log difference | 0 | Earnings and Productivity | Business Sector: Unit Labor Cost (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/ULCBS) |
| 141 | `ULCMFG` | 5 | First log difference | 1 | Earnings and Productivity | Manufacturing Sector: Unit Labor Cost (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/ULCMFG) |
| 142 | `ULCNFB` | 5 | First log difference | 1 | Earnings and Productivity | Nonfarm Business Sector: Unit Labor Cost (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/ULCNFB) |
| 143 | `UNLPNBS` | 5 | First log difference | 1 | Earnings and Productivity | Nonfarm Business Sector: Unit Nonlabor Payments (Index 2009=100) | [FRED](https://fred.stlouisfed.org/series/UNLPNBS) |
| 144 | `FEDFUNDS` | 2 | First difference | 1 | Interest Rates | Effective Federal Funds Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/FEDFUNDS) |
| 145 | `TB3MS` | 2 | First difference | 1 | Interest Rates | 3-Month Treasury Bill: Secondary Market Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/TB3MS) |
| 146 | `TB6MS` | 2 | First difference | 0 | Interest Rates | 6-Month Treasury Bill: Secondary Market Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/TB6MS) |
| 147 | `GS1` | 2 | First difference | 0 | Interest Rates | 1-Year Treasury Constant Maturity Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/GS1) |
| 148 | `GS10` | 2 | First difference | 0 | Interest Rates | 10-Year Treasury Constant Maturity Rate (Percent) | [FRED](https://fred.stlouisfed.org/series/GS10) |
| 149 | `MORTGAGE30US` | 2 | First difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/MORTGAGE30US) |
| 150 | `AAA` | 2 | First difference | 0 | Interest Rates | Moody’s Seasoned Aaa Corporate Bond Yield (Percent) | [FRED](https://fred.stlouisfed.org/series/AAA) |
| 151 | `BAA` | 2 | First difference | 0 | Interest Rates | Moody’s Seasoned Baa Corporate Bond Yield (Percent) | [FRED](https://fred.stlouisfed.org/series/BAA) |
| 152 | `BAA10YM` | 1 | No transformation | 1 | Interest Rates | Moody’s Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity (Percent) | [FRED](https://fred.stlouisfed.org/series/BAA10YM) |
| 153 | `MORTG10YRx` | 1 | No transformation | 1 | Interest Rates | 30-Year Conventional Mortgage Rate Relative to 10-Year Treasury Constant Maturity (Percent) | constructed / appendix |
| 154 | `TB6M3Mx` | 1 | No transformation | 1 | Interest Rates | 6-Month Treasury Bill Minus 3-Month Treasury Bill, secondary market (Percent) | constructed / appendix |
| 155 | `GS1TB3Mx` | 1 | No transformation | 1 | Interest Rates | 1-Year Treasury Constant Maturity Minus 3-Month Treasury Bill, secondary market (Percent) | constructed / appendix |
| 156 | `GS10TB3Mx` | 1 | No transformation | 1 | Interest Rates | 10-Year Treasury Constant Maturity Minus 3-Month Treasury Bill, secondary market (Percent) | constructed / appendix |
| 157 | `CPF3MTB3Mx` | 1 | No transformation | 1 | Interest Rates | CP Tbill Spread 3-Month Commercial Paper Minus 3-Month Treasury Bill, secondary market (Percent) | constructed / appendix |
| 158 | `BOGMBASEREALx` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 159 | `M1REAL` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/M1REAL) |
| 160 | `M2REAL` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/M2REAL) |
| 161 | `BUSLOANSx` | 5 | First log difference | 1 | Money and Credit | Real C&Lloand Real Commercial and Industrial Loans, All Commercial Banks (Billions of 2009 U.S. Dollars), deflated by Core PCE | constructed / appendix |
| 162 | `CONSUMERx` | 5 | First log difference | 1 | Money and Credit | Real Consumer Loans at All Commercial Banks (Billions of 2009 U.S. Dollars), deflated by Core PCE | constructed / appendix |
| 163 | `NONREVSLx` | 5 | First log difference | 1 | Money and Credit | Real NonRevCredit Total Real Nonrevolving Credit Owned and Securitized, Outstanding (Billions of Dollars), deflated by Core PCE | constructed / appendix |
| 164 | `REALLNx` | 5 | First log difference | 1 | Money and Credit | Real LoansRealEst Real Real Estate Loans, All Commercial Banks (Billions of 2009 U.S. Dollars), deflated by Core PCE | constructed / appendix |
| 165 | `REVOLSLx` | 5 | First log difference | 1 | Money and Credit | Real RevolvCredit Total Real Revolving Credit Owned and Securitized, Outstanding (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 166 | `TOTALSLx` | 5 | First log difference | 0 | Money and Credit | Total Consumer Credit Outstanding, deflated by Core PCE | constructed / appendix |
| 167 | `DRIWCIL` | 1 | No transformation | 1 | Money and Credit | FRBSLO Consumers FRB Senior Loans Officer Opions. Net Percentage of Domestic Respondents Reporting Increased Willingness to Make Consumer Installment Loans | [FRED](https://fred.stlouisfed.org/series/DRIWCIL) |
| 168 | `TABSHNOx` | 5 | First log difference | 0 | Household Balance Sheets | Real HHW:TASAReal Total Assets of Households and Nonprofit Organizations (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 169 | `TLBSHNOx` | 5 | First log difference | 1 | Household Balance Sheets | Real Total Liabilities of Households and Nonprofit Organizations (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 170 | `LIABPIx` | 5 | First log difference | 0 | Household Balance Sheets | Liabilities of Households and Nonprofit Organizations Relative to Personal Disposable Income (Percent) | constructed / appendix |
| 171 | `TNWBSHNOx` | 5 | First log difference | 1 | Household Balance Sheets | Real Net Worth of Households and Nonprofit Organizations (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 172 | `NWPIx` | 1 | No transformation | 0 | Household Balance Sheets | Net Worth of Households and Nonprofit Organizations Relative to Disposable Personal Income (Percent) | constructed / appendix |
| 173 | `TARESAx` | 5 | First log difference | 1 | Household Balance Sheets | Real HHW:TA RESA Real Assets of Households and Nonprofit Organizations excluding Real Estate Assets (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 174 | `HNOREMQ027Sx` | 5 | First log difference | 1 | Household Balance Sheets | Real Real Estate Assets of Households and Nonprofit Organizations (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 175 | `TFAABSHNOx` | 5 | First log difference | 1 | Household Balance Sheets | Real Total Financial Assets of Households and Nonprofit Organizations (Billions of 2009 Dollars), deflated by Core PCE | constructed / appendix |
| 176 | `VIXCLSx` | 1 | No transformation | 1 | - | See official appendix / FRED source page. | constructed / appendix |
| 177 | `USSTHPI` | 5 | First log difference | 1 | Housing | Real Hprice:OFHEO All-Transactions House Price Index for the United States (Index 1980 Q1=100) | [FRED](https://fred.stlouisfed.org/series/USSTHPI) |
| 178 | `SPCS10RSA` | 5 | First log difference | 1 | Housing | S&P/Case-Shiller 10-City Composite Home Price Index (Index January 2000 = 100) | [FRED](https://fred.stlouisfed.org/series/SPCS10RSA) |
| 179 | `SPCS20RSA` | 5 | First log difference | 1 | Housing | S&P/Case-Shiller 20-City Composite Home Price Index (Index January 2000 = 100) | [FRED](https://fred.stlouisfed.org/series/SPCS20RSA) |
| 180 | `TWEXAFEGSMTHx` | 5 | First log difference | 1 | - | See official appendix / FRED source page. | constructed / appendix |
| 181 | `EXUSEU` | 5 | First log difference | 1 | Exchange Rates | U.S. / Euro Foreign Exchange Rate (U.S. Dollars to One Euro) | [FRED](https://fred.stlouisfed.org/series/EXUSEU) |
| 182 | `EXSZUSx` | 5 | First log difference | 1 | Exchange Rates | Switzerland / U.S. Foreign Exchange Rate | constructed / appendix |
| 183 | `EXJPUSx` | 5 | First log difference | 1 | Exchange Rates | Japan / U.S. Foreign Exchange Rate | constructed / appendix |
| 184 | `EXUSUKx` | 5 | First log difference | 1 | Exchange Rates | U.S. / U.K. Foreign Exchange Rate | constructed / appendix |
| 185 | `EXCAUSx` | 5 | First log difference | 1 | Exchange Rates | Canada / U.S. Foreign Exchange Rate | constructed / appendix |
| 186 | `UMCSENTx` | 1 | No transformation | 1 | Other | Cons. Expectations University of Michigan: Consumer Sentiment (Index 1st Quarter 1966=100) | constructed / appendix |
| 187 | `USEPUINDXM` | 2 | First difference | 1 | Other | Economic Policy Uncertainty Index for United States | [FRED](https://fred.stlouisfed.org/series/USEPUINDXM) |
| 188 | `B020RE1Q156NBEA` | 2 | First difference | 0 | NIPA | Shares of gross domestic product: Exports of goods and services (Percent) | [FRED](https://fred.stlouisfed.org/series/B020RE1Q156NBEA) |
| 189 | `B021RE1Q156NBEA` | 2 | First difference | 0 | NIPA | Shares of gross domestic product: Imports of goods and services (Percent) | [FRED](https://fred.stlouisfed.org/series/B021RE1Q156NBEA) |
| 190 | `GFDEGDQ188S` | 2 | First difference | 0 | Non-Household Balance Sheets | Federal Debt: Total Public Debt as Percent of GDP (Percent) | [FRED](https://fred.stlouisfed.org/series/GFDEGDQ188S) |
| 191 | `GFDEBTNx` | 2 | First difference | 0 | Non-Household Balance Sheets | Real Federal Debt: Total Public Debt (Millions of 2009 Dollars), deflated by PCE | constructed / appendix |
| 192 | `IPMANSICS` | 5 | First log difference | 0 | Industrial Production | Industrial Production: Manufacturing (SIC) (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPMANSICS) |
| 193 | `IPB51222S` | 5 | First log difference | 0 | Industrial Production | Industrial Production: Residential Utilities (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPB51222S) |
| 194 | `IPFUELS` | 5 | First log difference | 0 | Industrial Production | Industrial Production: Fuels (Index 2012=100) | [FRED](https://fred.stlouisfed.org/series/IPFUELS) |
| 195 | `UEMPMEAN` | 2 | First difference | 0 | Employment and Unemployment, continued | Average (Mean) Duration of Unemployment (Weeks) | [FRED](https://fred.stlouisfed.org/series/UEMPMEAN) |
| 196 | `CES0600000007` | 2 | First difference | 0 | Employment and Unemployment, continued | Average Weekly Hours of Production and Nonsupervisory Employees: Goods-Producing | [FRED](https://fred.stlouisfed.org/series/CES0600000007) |
| 197 | `TOTRESNS` | 6 | Second log difference | 0 | Money and Credit | Total Reserves of Depository Institutions (Billions of Dollars) | [FRED](https://fred.stlouisfed.org/series/TOTRESNS) |
| 198 | `NONBORRES` | 7 | First difference of percent change | 0 | Money and Credit | Reserves Of Depository Institutions, Nonborrowed (Millions of Dollars) | [FRED](https://fred.stlouisfed.org/series/NONBORRES) |
| 199 | `GS5` | 2 | First difference | 0 | Interest Rates | 5-Year Treasury Constant Maturity Rate | [FRED](https://fred.stlouisfed.org/series/GS5) |
| 200 | `TB3SMFFM` | 1 | No transformation | 0 | Interest Rates | 3-Month Treasury Constant Maturity Minus Federal Funds Rate | [FRED](https://fred.stlouisfed.org/series/TB3SMFFM) |
| 201 | `T5YFFM` | 1 | No transformation | 0 | Interest Rates | 5-Year Treasury Constant Maturity Minus Federal Funds Rate | [FRED](https://fred.stlouisfed.org/series/T5YFFM) |
| 202 | `AAAFFM` | 1 | No transformation | 0 | Interest Rates | Moody’s Seasoned Aaa Corporate Bond Minus Federal Funds Rate | [FRED](https://fred.stlouisfed.org/series/AAAFFM) |
| 203 | `WPSID62` | 6 | Second log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/WPSID62) |
| 204 | `PPICMM` | 6 | Second log difference | 0 | Prices, continued | Producer Price Index: Commodities: Metals and metal products: Primary nonferrous metals (Index 1982=100) | [FRED](https://fred.stlouisfed.org/series/PPICMM) |
| 205 | `CPIAPPSL` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: Apparel (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CPIAPPSL) |
| 206 | `CPITRNSL` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: Transportation (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CPITRNSL) |
| 207 | `CPIMEDSL` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: Medical Care (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CPIMEDSL) |
| 208 | `CUSR0000SAC` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: Commodities (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CUSR0000SAC) |
| 209 | `CUSR0000SAD` | 6 | Second log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/CUSR0000SAD) |
| 210 | `CUSR0000SAS` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: Services (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CUSR0000SAS) |
| 211 | `CPIULFSL` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: All Items Less Food (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CPIULFSL) |
| 212 | `CUSR0000SA0L2` | 6 | Second log difference | 0 | - | See official appendix / FRED source page. | [FRED](https://fred.stlouisfed.org/series/CUSR0000SA0L2) |
| 213 | `CUSR0000SA0L5` | 6 | Second log difference | 0 | Prices, continued | Consumer Price Index for All Urban Consumers: All items less medical care (Index 1982-84=100) | [FRED](https://fred.stlouisfed.org/series/CUSR0000SA0L5) |
| 214 | `CES0600000008` | 6 | Second log difference | 0 | Earnings and Productivity | Average Hourly Earnings of Production and Nonsupervisory Employees: Goods-Producing (Dollars per Hour) | [FRED](https://fred.stlouisfed.org/series/CES0600000008) |
| 215 | `DTCOLNVHFNM` | 6 | Second log difference | 0 | Money and Credit | Consumer Motor Vehicle Loans Outstanding Owned by Finance Companies (Millions of Dollars) | [FRED](https://fred.stlouisfed.org/series/DTCOLNVHFNM) |
| 216 | `DTCTHFNM` | 6 | Second log difference | 0 | Money and Credit | Total Consumer Loans and Leases Outstanding Owned and Securitized by Finance Companies (Millions of Dollars) | [FRED](https://fred.stlouisfed.org/series/DTCTHFNM) |
| 217 | `INVEST` | 6 | Second log difference | 0 | Money and Credit | Securities in Bank Credit at All Commercial Banks (Billions of Dollars) | [FRED](https://fred.stlouisfed.org/series/INVEST) |
| 218 | `HWIURATIOx` | 2 | First difference | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 219 | `CLAIMSx` | 5 | First log difference | 0 | Employment and Unemployment, continued | Initial Claims | constructed / appendix |
| 220 | `BUSINVx` | 5 | First log difference | 0 | Inventories, Orders, and Sales | Total Business Inventories (Millions of Dollars) | constructed / appendix |
| 221 | `ISRATIOx` | 2 | First difference | 0 | Inventories, Orders, and Sales | Total Business: Inventories to Sales Ratio | constructed / appendix |
| 222 | `CONSPIx` | 2 | First difference | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 223 | `CP3M` | 2 | First difference | 0 | Interest Rates | 3-Month AA Financial Commercial Paper Rate | [FRED](https://fred.stlouisfed.org/series/CP3M) |
| 224 | `COMPAPFF` | 1 | No transformation | 0 | Interest Rates | 3-Month Commercial Paper Minus Federal Funds Rate | [FRED](https://fred.stlouisfed.org/series/COMPAPFF) |
| 225 | `PERMITNE` | 5 | First log difference | 0 | Housing | New Private Housing Units Authorized by Building Permits in the Northeast Census Region (Thousands, SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITNE) |
| 226 | `PERMITMW` | 5 | First log difference | 0 | Housing | New Private Housing Units Authorized by Building Permits in the Midwest Census Region (Thousands, SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITMW) |
| 227 | `PERMITS` | 5 | First log difference | 0 | Housing | New Private Housing Units Authorized by Building Permits in the South Census Region (Thousands, SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITS) |
| 228 | `PERMITW` | 5 | First log difference | 0 | Housing | New Private Housing Units Authorized by Building Permits in the West Census Region (Thousands, SAAR) | [FRED](https://fred.stlouisfed.org/series/PERMITW) |
| 229 | `NIKKEI225` | 5 | First log difference | 0 | Stock Markets | Nikkei Stock Average | [FRED](https://fred.stlouisfed.org/series/NIKKEI225) |
| 230 | `NASDAQCOM` | 5 | First log difference | 0 | Stock Markets | NASDAQ Composite (Index Feb 5, 1971=100) | [FRED](https://fred.stlouisfed.org/series/NASDAQCOM) |
| 231 | `CUSR0000SEHC` | 6 | Second log difference | 0 | Prices, continued | CPI for All Urban Consumers: Owners’ equivalent rent of residences (Index Dec 1982=100) | [FRED](https://fred.stlouisfed.org/series/CUSR0000SEHC) |
| 232 | `TLBSNNCBx` | 5 | First log difference | 0 | Non-Household Balance Sheets | Real Nonfinancial Corporate Business Sector Liabilities (Billions of 2009 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS | constructed / appendix |
| 233 | `TLBSNNCBBDIx` | 1 | No transformation | 0 | Non-Household Balance Sheets | Nonfinancial Corporate Business Sector Liabilities to Disposable Business Income (Percent) | constructed / appendix |
| 234 | `TTAABSNNCBx` | 5 | First log difference | 0 | Non-Household Balance Sheets | Real Nonfinancial Corporate Business Sector Assets (Billions of 2009 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS | constructed / appendix |
| 235 | `TNWMVBSNNCBx` | 5 | First log difference | 0 | Non-Household Balance Sheets | Real Nonfinancial Corporate Business Sector Net Worth (Billions of 2009 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS | constructed / appendix |
| 236 | `TNWMVBSNNCBBDIx` | 2 | First difference | 0 | Non-Household Balance Sheets | Nonfinancial Corporate Business Sector Net Worth to Disposable Business Income (Percent) | constructed / appendix |
| 237 | `TLBSNNBx` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 238 | `TLBSNNBBDIx` | 1 | No transformation | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 239 | `TABSNNBx` | 5 | First log difference | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 240 | `TNWBSNNBx` | 5 | First log difference | 0 | Non-Household Balance Sheets | Real Nonfinancial Noncorporate Business Sector Net Worth (Billions of 2009 Dollars), Deflated by Implicit Price Deflator for Business Sector IPDBS | constructed / appendix |
| 241 | `TNWBSNNBBDIx` | 2 | First difference | 0 | Non-Household Balance Sheets | Nonfinancial Noncorporate Business Sector Net Worth to Disposable Business Income (Percent) | constructed / appendix |
| 242 | `CNCFx` | 5 | First log difference | 0 | Non-Household Balance Sheets | Real Disposable Business Income, Billions of 2009 Dollars (Corporate cash flow with IVA minus taxes on corporate income, deflated by Implicit Price Deflator for Business Sector IPDBS) | constructed / appendix |
| 243 | `S&P 500` | 5 | First log difference | 1 | Stock Markets | S&P’s Common Stock Price Index: Composite | constructed / appendix |
| 244 | `S&P div yield` | 2 | First difference | 0 | - | See official appendix / FRED source page. | constructed / appendix |
| 245 | `S&P PE ratio` | 5 | First log difference | 0 | Stock Markets | S&P’s Composite Common Stock: Price-Earnings Ratio | constructed / appendix |
