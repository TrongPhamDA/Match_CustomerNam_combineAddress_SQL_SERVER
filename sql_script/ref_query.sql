WITH cust_raw AS
    (
    SELECT
        CAST(c.REPORT_MONTH AS DATE)             REPORT_MONTH
       ,c.CUSTOMER_CD                            CUSTOMER_CODE
       ,c.CUST_NAME                              CUSTOMER_NAME
       ,CAST(c.DISTRIBUTOR_CD AS NVARCHAR (MAX)) DISTRIBUTOR_CODE
       ,c.LATITUDE                               CUSTOMER_LAT
       ,c.LONGITUDE                              CUSTOMER_LONG
       ,c.ADDR_1
       ,c.ADDR_2
       ,c.ADDR_3
       ,c.ADDR_4
       ,c.ADDR_5
    FROM    EDW.DM.VW_DIM_CUSTOMER_MONTHLY c
    )
    ,cust_latest AS
    (
    SELECT
        c.*
    FROM    cust_raw c
    WHERE   c.REPORT_MONTH = (SELECT    MAX(c_.REPORT_MONTH) LATEST_MONTH FROM  cust_raw c_)
    )
    ,dist AS
    (
    SELECT
        d.DISTRIBUTOR_CD DISTRIBUTOR_CODE
       ,d.GBA
    FROM    EDW.DM.VW_DIM_DISTRIBUTOR_LATEST d
    )
SELECT  DISTINCT
        c.CUSTOMER_CODE
       ,c.CUSTOMER_NAME
       ,c.CUSTOMER_LAT
       ,c.CUSTOMER_LONG
       ,COALESCE(c.ADDR_1, N'') + N' ' + COALESCE(c.ADDR_2, N'') + N' ' + COALESCE(c.ADDR_3, N'') + N' '
        + COALESCE(c.ADDR_4, N'') + N' ' + COALESCE(c.ADDR_5, N'') CUSTOMER_ADDRESS
FROM    cust_latest c
LEFT JOIN dist      d
       ON c.DISTRIBUTOR_CODE = d.DISTRIBUTOR_CODE
WHERE   d.GBA = 'GT'
        AND COALESCE(c.CUSTOMER_LAT, 0) NOT IN (1, 0)
        AND COALESCE(c.CUSTOMER_LONG, 0) NOT IN (1, 0)
ORDER BY c.CUSTOMER_LAT	 -- Bắc vào Nam   Việt Nam
        ,c.CUSTOMER_LONG -- Tây sang Đông Việt Nam