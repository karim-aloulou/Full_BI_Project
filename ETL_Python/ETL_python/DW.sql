USE [master]
GO
/****** Object:  Database [DW_Projet_py]    Script Date: 13/11/2022 17:35:14 ******/
CREATE DATABASE [DW_Projet_py]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DW_Projet_py', FILENAME = N'C:\Program Files (x86)\Microsoft SQL Server\MSSQL11.MSSQLSERVER\MSSQL\DATA\DW_Projet_py.mdf' , SIZE = 7168KB , MAXSIZE = UNLIMITED, FILEGROWTH = 1024KB )
 LOG ON 
( NAME = N'DW_Projet_py_log', FILENAME = N'C:\Program Files (x86)\Microsoft SQL Server\MSSQL11.MSSQLSERVER\MSSQL\DATA\DW_Projet_py_log.ldf' , SIZE = 5696KB , MAXSIZE = 2048GB , FILEGROWTH = 10%)
GO
ALTER DATABASE [DW_Projet_py] SET COMPATIBILITY_LEVEL = 110
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DW_Projet_py].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DW_Projet_py] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DW_Projet_py] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DW_Projet_py] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DW_Projet_py] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [DW_Projet_py] SET ARITHABORT OFF 
GO
ALTER DATABASE [DW_Projet_py] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [DW_Projet_py] SET AUTO_CREATE_STATISTICS ON 
GO
ALTER DATABASE [DW_Projet_py] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [DW_Projet_py] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [DW_Projet_py] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [DW_Projet_py] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [DW_Projet_py] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [DW_Projet_py] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [DW_Projet_py] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [DW_Projet_py] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [DW_Projet_py] SET  DISABLE_BROKER 
GO
ALTER DATABASE [DW_Projet_py] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [DW_Projet_py] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [DW_Projet_py] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [DW_Projet_py] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [DW_Projet_py] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [DW_Projet_py] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [DW_Projet_py] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [DW_Projet_py] SET RECOVERY FULL 
GO
ALTER DATABASE [DW_Projet_py] SET  MULTI_USER 
GO
ALTER DATABASE [DW_Projet_py] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [DW_Projet_py] SET DB_CHAINING OFF 
GO
ALTER DATABASE [DW_Projet_py] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [DW_Projet_py] SET TARGET_RECOVERY_TIME = 0 SECONDS 
GO
USE [DW_Projet_py]
GO
/****** Object:  Table [dbo].[dimCustomer]    Script Date: 13/11/2022 17:35:14 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[dimCustomer](
	[Customer_PK] [int] NOT NULL,
	[Customer_ID] [varchar](255) NULL,
	[Segment] [varchar](255) NULL,
 CONSTRAINT [PK_dimCustomer2] PRIMARY KEY CLUSTERED 
(
	[Customer_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[DimDate]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[DimDate](
	[DateKey] [int] NOT NULL,
	[Date] [date] NULL,
	[FullDate] [varchar](2) NULL,
	[DayOfMonth] [varchar](2) NULL,
	[DayName] [varchar](9) NULL,
	[DayOfWeek] [char](1) NULL,
	[DayOfWeekInMonth] [varchar](2) NULL,
	[DayOfWeekInYear] [varchar](2) NULL,
	[DayOfQuarter] [varchar](3) NULL,
	[DayOfYear] [varchar](3) NULL,
	[WeekOfMonth] [varchar](1) NULL,
	[WeekOfQuarter] [varchar](2) NULL,
	[WeekOfYear] [varchar](2) NULL,
	[Month] [varchar](2) NULL,
	[MonthName] [varchar](9) NULL,
	[MonthOfQuarter] [varchar](2) NULL,
	[Quarter] [char](1) NULL,
	[QuarterName] [varchar](9) NULL,
	[Year] [char](4) NULL,
	[YearName] [char](7) NULL,
	[MonthYear] [char](10) NULL,
	[MMYYYY] [char](6) NULL,
	[FirstDayOfMonth] [date] NULL,
	[LastDayOfMonth] [date] NULL,
	[FirstDayOfQuarter] [date] NULL,
	[LastDayOfQuarter] [date] NULL,
	[FirstDayOfYear] [date] NULL,
	[LastDayOfYear] [date] NULL,
 CONSTRAINT [PK__DimDate__40DF45E3AAB86ADA] PRIMARY KEY CLUSTERED 
(
	[DateKey] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[DimLocalisation]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[DimLocalisation](
	[Localisation_PK] [int] NOT NULL,
	[City] [varchar](255) NULL,
	[State] [varchar](255) NULL,
	[Country] [varchar](255) NULL,
	[Region] [varchar](255) NULL,
	[Market] [varchar](255) NULL,
	[Postal_code] [varchar](255) NULL,
 CONSTRAINT [PK_DimLocalisation] PRIMARY KEY CLUSTERED 
(
	[Localisation_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[DimOrder]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[DimOrder](
	[Order_PK] [int] NOT NULL,
	[Order_Priority] [varchar](255) NULL,
	[Order_ID] [varchar](255) NULL,
 CONSTRAINT [PK_DimOrder] PRIMARY KEY CLUSTERED 
(
	[Order_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[DimProduct]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[DimProduct](
	[Product_PK] [int] NOT NULL,
	[Product_ID] [varchar](255) NULL,
	[Category] [varchar](255) NULL,
	[Sub_Category] [varchar](255) NULL,
	[Product_Name] [varchar](255) NULL,
 CONSTRAINT [PK_DimProduct] PRIMARY KEY CLUSTERED 
(
	[Product_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[DimShip]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[DimShip](
	[Ship_PK] [int] IDENTITY(1,1) NOT NULL,
	[Ship_Code] [bigint] NULL,
	[Ship_Mode] [varchar](255) NULL,
 CONSTRAINT [PK_DimShip] PRIMARY KEY CLUSTERED 
(
	[Ship_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[DimShip0]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [dbo].[DimShip0](
	[Ship_PK] [int] NOT NULL,
	[Ship_Code] [bigint] NULL,
	[Ship_Mode] [varchar](255) NULL,
 CONSTRAINT [PK_DimShip0] PRIMARY KEY CLUSTERED 
(
	[Ship_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [dbo].[FactSales]    Script Date: 13/11/2022 17:35:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FactSales](
	[FK_Customer] [int] NOT NULL,
	[FK_Product] [int] NOT NULL,
	[FK_Ship] [int] NOT NULL,
	[FK_Date_Order] [int] NOT NULL,
	[FK_Date_Ship] [int] NOT NULL,
	[FK_Order] [int] NOT NULL,
	[FK_Localisation] [int] NOT NULL,
	[Quantity] [int] NULL,
	[Profit] [float] NULL,
	[Sales] [float] NULL,
	[Discount] [float] NULL,
	[Returned] [nvarchar](255) NULL,
	[Ship_Coast] [float] NULL,
	[Ship_Duration] [int] NULL,
 CONSTRAINT [PK_FactSales] PRIMARY KEY CLUSTERED 
(
	[FK_Customer] ASC,
	[FK_Product] ASC,
	[FK_Ship] ASC,
	[FK_Date_Order] ASC,
	[FK_Date_Ship] ASC,
	[FK_Order] ASC,
	[FK_Localisation] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_dimCustomer] FOREIGN KEY([FK_Customer])
REFERENCES [dbo].[dimCustomer] ([Customer_PK])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_dimCustomer]
GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_DimDate_Order] FOREIGN KEY([FK_Date_Order])
REFERENCES [dbo].[DimDate] ([DateKey])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_DimDate_Order]
GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_DimDate_Ship] FOREIGN KEY([FK_Date_Ship])
REFERENCES [dbo].[DimDate] ([DateKey])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_DimDate_Ship]
GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_DimLocalisation] FOREIGN KEY([FK_Localisation])
REFERENCES [dbo].[DimLocalisation] ([Localisation_PK])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_DimLocalisation]
GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_DimOrder] FOREIGN KEY([FK_Order])
REFERENCES [dbo].[DimOrder] ([Order_PK])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_DimOrder]
GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_DimProduct] FOREIGN KEY([FK_Product])
REFERENCES [dbo].[DimProduct] ([Product_PK])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_DimProduct]
GO
ALTER TABLE [dbo].[FactSales]  WITH CHECK ADD  CONSTRAINT [FK_FactSales_DimShip] FOREIGN KEY([FK_Ship])
REFERENCES [dbo].[DimShip] ([Ship_PK])
GO
ALTER TABLE [dbo].[FactSales] CHECK CONSTRAINT [FK_FactSales_DimShip]
GO
USE [master]
GO
ALTER DATABASE [DW_Projet_py] SET  READ_WRITE 
GO
