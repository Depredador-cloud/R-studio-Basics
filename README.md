# R-studio-Basics

### 🚀 Getting Started with R: A Basic Guide

Welcome to the world of R! R is a powerful language for statistical computing and graphics. It's widely used among statisticians, data miners, and data scientists for data analysis and developing statistical software. This guide will take you through the basics of R and RStudio, helping you get started with ease.

---

#### 📥 Installation
1. **Download R**: Visit [CRAN](https://cran.r-project.org/) and download the version suitable for your operating system.
2. **Download RStudio**: Go to [RStudio](https://www.rstudio.com/products/rstudio/download/) and download the free version.

---

#### 🔍 RStudio Interface Overview
RStudio provides a user-friendly interface to work with R. Here's a quick tour:
- **Console**: This is where you can type R commands and see immediate results.
- **Source Pane**: Write, edit, and save R scripts here.
- **Environment/History Pane**: View and manage your variables and command history.
- **Files/Plots/Packages/Help Pane**: Manage files, view plots, load packages, and access help.

![RStudio Interface](https://www.rstudio.com/wp-content/uploads/2016/10/rstudio-ide.png)

---

#### 📈 Basic R Syntax
R is case-sensitive and has a simple syntax. Let's start with some basics:

##### Variables and Data Types
```r
# Assigning values
x <- 10
y <- 20

# Basic operations
sum <- x + y
diff <- y - x

# Print values
print(sum)
print(diff)
```

##### Vectors and Factors
Vectors are sequences of data elements of the same type.
```r
# Creating a vector
vec <- c(1, 2, 3, 4, 5)

# Accessing elements
print(vec[1])  # First element
print(vec[1:3])  # First three elements
```

Factors are used to represent categorical data.
```r
# Creating a factor
factor_vec <- factor(c("Male", "Female", "Female", "Male"))

# Summary of the factor
summary(factor_vec)
```

##### Data Frames
Data frames are table-like structures.
```r
# Creating a data frame
df <- data.frame(
  Name = c("John", "Jane", "Doe"),
  Age = c(23, 25, 28)
)

# Accessing data
print(df$Name)  # Column access
print(df[1, ])  # Row access
```

---

#### 🛠️ Extensive List of R Libraries and Their Uses

R has a rich ecosystem of libraries for various data science tasks. Here is an extensive list of some essential libraries and their uses:

##### **Data Manipulation**
- **dplyr**: A grammar of data manipulation, providing a consistent set of verbs for data manipulation.
  ```r
  library(dplyr)
  df %>% filter(Age > 25) %>% summarize(Average_Age = mean(Age))
  ```
- **tidyr**: Tools for tidying messy data.
  ```r
  library(tidyr)
  df <- df %>% gather(key, value, -Name)
  ```

##### **Data Import**
- **readr**: Fast and friendly way to read rectangular data.
  ```r
  library(readr)
  df <- read_csv("data.csv")
  ```
- **data.table**: Fast aggregation of large data (e.g., 100GB in RAM), fast ordered joins, fast add/modify/delete of columns by group, and fast file reads.
  ```r
  library(data.table)
  df <- fread("data.csv")
  ```

##### **Data Visualization**
- **ggplot2**: A system for declaratively creating graphics, based on The Grammar of Graphics.
  ```r
  library(ggplot2)
  ggplot(df, aes(x = Name, y = Age)) + geom_point()
  ```
- **lattice**: An implementation of Trellis graphics for R.
  ```r
  library(lattice)
  xyplot(Age ~ Name, data = df)
  ```
- **plotly**: Create interactive web graphics via the open source JavaScript graphing library plotly.js.
  ```r
  library(plotly)
  plot_ly(df, x = ~Name, y = ~Age, type = 'scatter', mode = 'markers')
  ```

##### **Time Series Analysis**
- **xts**: eXtensible time series.
  ```r
  library(xts)
  ts <- xts(data, order.by = as.Date(dates))
  ```
- **zoo**: S3 Infrastructure for Regular and Irregular Time Series (Z’s ordered observations).
  ```r
  library(zoo)
  ts <- zoo(data, order.by = as.Date(dates))
  ```

##### **Machine Learning**
- **caret**: Classification and Regression Training.
  ```r
  library(caret)
  model <- train(Species ~ ., data = iris, method = "rf")
  ```
- **randomForest**: Breiman and Cutler's Random Forests for Classification and Regression.
  ```r
  library(randomForest)
  model <- randomForest(Species ~ ., data = iris)
  ```

##### **Text Mining**
- **tm**: Text Mining Package.
  ```r
  library(tm)
  corpus <- Corpus(VectorSource(text))
  ```
- **quanteda**: Quantitative analysis of textual data.
  ```r
  library(quanteda)
  dfm <- dfm(text)
  ```

##### **Web Scraping**
- **rvest**: Simple web scraping for R.
  ```r
  library(rvest)
  webpage <- read_html("http://example.com")
  ```

##### **Geospatial Analysis**
- **sf**: Simple Features for R.
  ```r
  library(sf)
  sf_data <- st_read("data.shp")
  ```
- **sp**: Classes and methods for spatial data.
  ```r
  library(sp)
  sp_data <- readOGR("data.shp")
  ```

##### **Reporting**
- **knitr**: A general-purpose tool for dynamic report generation in R.
  ```r
  library(knitr)
  knit("report.Rmd")
  ```
- **rmarkdown**: Dynamic documents for R.
  ```r
  library(rmarkdown)
  render("report.Rmd")
  ```

##### **Interactive Dashboards**
- **shiny**: Easy interactive web applications with R.
  ```r
  library(shiny)
  ui <- fluidPage(
    titlePanel("Hello Shiny!"),
    sidebarLayout(
      sidebarPanel(
        sliderInput("obs", "Number of observations:", 1, 100, 50)
      ),
      mainPanel(
        plotOutput("distPlot")
      )
    )
  )
  server <- function(input, output) {
    output$distPlot <- renderPlot({
      hist(rnorm(input$obs))
    })
  }
  shinyApp(ui = ui, server = server)
  ```

##### **Bioinformatics**
- **Bioconductor**: Provides tools for the analysis and comprehension of high-throughput genomic data.
  ```r
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("GenomicFeatures")
  library(GenomicFeatures)
  ```

#### 📚 Extensive List of R Libraries and Their Uses (Continued)

##### **Statistical Analysis**
- **MASS**: Functions and datasets to support Venables and Ripley, "Modern Applied Statistics with S" (4th edition, 2002).
  ```r
  library(MASS)
  fit <- lm(mpg ~ wt + qsec + drat, data = mtcars)
  ```
- **car**: Companion to Applied Regression.
  ```r
  library(car)
  Anova(fit)
  ```

##### **Econometrics**
- **plm**: Linear models for panel data.
  ```r
  library(plm)
  panel_model <- plm(y ~ x1 + x2, data = panel_data, index = c("id", "time"))
  ```
- **forecast**: Forecasting functions for time series and linear models.
  ```r
  library(forecast)
  fit <- auto.arima(ts_data)
  forecast(fit, h = 10)
  ```

##### **Data Import/Export**
- **haven**: Import and export SPSS, Stata, and SAS files.
  ```r
  library(haven)
  df <- read_spss("data.sav")
  ```
- **openxlsx**: Read, write, and edit Excel files.
  ```r
  library(openxlsx)
  write.xlsx(df, "data.xlsx")
  ```

##### **Data Cleaning**
- **janitor**: Simple tools for data cleaning.
  ```r
  library(janitor)
  df <- clean_names(df)
  ```
- **assertr**: Assertive programming for R.
  ```r
  library(assertr)
  verify(df, within_bounds(0, 100), column_name)
  ```

##### **Network Analysis**
- **igraph**: Network analysis and visualization.
  ```r
  library(igraph)
  g <- graph_from_data_frame(d, directed = FALSE)
  plot(g)
  ```
- **networkD3**: D3 JavaScript Network Graphs from R.
  ```r
  library(networkD3)
  simpleNetwork(Data = d)
  ```

##### **Biostatistics**
- **survival**: Survival analysis.
  ```r
  library(survival)
  fit <- survfit(Surv(time, status) ~ x, data = df)
  plot(fit)
  ```
- **survminer**: Drawing Survival Curves using 'ggplot2'.
  ```r
  library(survminer)
  ggsurvplot(fit, data = df)
  ```

##### **Interactive Visualization**
- **highcharter**: A wrapper for the Highcharts library.
  ```r
  library(highcharter)
  hchart(df, "line", hcaes(x = x, y = y))
  ```
- **leaflet**: Interactive maps.
  ```r
  library(leaflet)
  leaflet() %>% addTiles() %>% addMarkers(lng = -93.65, lat = 42.0285, popup = "Hello R!")
  ```

##### **Parallel Computing**
- **parallel**: Support for parallel computation using multiple processes.
  ```r
  library(parallel)
  cl <- makeCluster(detectCores() - 1)
  parLapply(cl, 1:10, function(x) x^2)
  stopCluster(cl)
  ```
- **foreach**: Provides support for the foreach looping construct.
  ```r
  library(foreach)
  foreach(i = 1:10, .combine = c) %dopar% (i^2)
  ```

##### **Database Interaction**
- **DBI**: Database interface.
  ```r
  library(DBI)
  con <- dbConnect(RSQLite::SQLite(), ":memory:")
  dbWriteTable(con, "mtcars", mtcars)
  dbGetQuery(con, "SELECT * FROM mtcars")
  dbDisconnect(con)
  ```
- **RMySQL**: MySQL database interface.
  ```r
  library(RMySQL)
  con <- dbConnect(MySQL(), user = 'username', password = 'password', dbname = 'database', host = 'host')
  dbGetQuery(con, "SELECT * FROM table")
  dbDisconnect(con)
  ```

##### **Spatial Analysis**
- **rgdal**: Bindings for the 'Geospatial' Data Abstraction Library.
  ```r
  library(rgdal)
  shp <- readOGR("path/to/shapefile.shp")
  ```
- **rgeos**: Interface to Geometry Engine - Open Source (GEOS).
  ```r
  library(rgeos)
  gArea(shp)
  ```

##### **Genetics**
- **genetics**: Classes and methods for handling genetic data.
  ```r
  library(genetics)
  geno <- genotype(c("AA", "AB", "BB"))
  summary(geno)
  ```
- **qtl**: Tools for analyzing QTL experiments.
  ```r
  library(qtl)
  data(fake.f2)
  summary(fake.f2)
  ```

##### **Financial Analysis**
- **quantmod**: Quantitative financial modelling framework.
  ```r
  library(quantmod)
  getSymbols("AAPL")
  chartSeries(AAPL)
  ```
- **PerformanceAnalytics**: Econometric tools for performance and risk analysis.
  ```r
  library(PerformanceAnalytics)
  charts.PerformanceSummary(df)
  ```

##### **Unit Testing**
- **testthat**: Unit testing for R.
  ```r
  library(testthat)
  test_that("multiplication works", {
    expect_equal(2 * 2, 4)
  })
  ```

##### **Web Development**
- **plumber**: Convert your R code into a web API.
  ```r
  library(plumber)
  r <- plumb("path/to/file.R")
  r$run(port=8000)
  ```
- **blogdown**: Create blogs and websites with R Markdown.
  ```r
  library(blogdown)
  blogdown::new_site()
  ```
#### 📚 Extensive List of R Libraries and Their Uses (Continued)

##### **Deep Learning**
- **keras**: R interface to Keras, a high-level neural networks API.
  ```r
  library(keras)
  model <- keras_model_sequential() %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  ```
- **tensorflow**: Interface to TensorFlow.
  ```r
  library(tensorflow)
  tf$constant("Hello, TensorFlow!")
  ```

##### **Natural Language Processing (NLP)**
- **text**: A suite of tools for text analysis.
  ```r
  library(text)
  words <- text_tokens("Hello world")
  ```
- **tidytext**: Text mining using tidy data principles.
  ```r
  library(tidytext)
  tidy_words <- unnest_tokens(df, word, text_column)
  ```

##### **Simulation**
- **SimDesign**: Provides a framework for designing and running simulation studies.
  ```r
  library(SimDesign)
  results <- runSimulation(design = design, replications = 1000, generate = generate, analyse = analyse, summarise = summarise)
  ```

##### **Bayesian Analysis**
- **rstan**: R interface to Stan, a package for Bayesian inference.
  ```r
  library(rstan)
  fit <- stan(file = 'model.stan', data = data, iter = 1000, chains = 4)
  ```
- **brms**: Bayesian regression models using Stan.
  ```r
  library(brms)
  fit <- brm(y ~ x1 + x2, data = df, family = "gaussian")
  ```

##### **Optimization**
- **optimx**: Extended versions of the 'optim' function to unify optimization functions.
  ```r
  library(optimx)
  result <- optimx(par = c(0,0), fn = fn, gr = gr, method = "L-BFGS-B")
  ```
- **nloptr**: Interface to NLopt, a library for nonlinear optimization.
  ```r
  library(nloptr)
  result <- nloptr(x0 = c(1,1), eval_f = eval_f, lb = c(0,0), ub = c(5,5), opts = list("algorithm" = "NLOPT_LD_MMA"))
  ```

##### **Graphical User Interface (GUI)**
- **tcltk**: Interface to Tcl/Tk for building GUIs.
  ```r
  library(tcltk)
  tt <- tktoplevel()
  label <- tklabel(tt, text = "Hello, world!")
  tkpack(label)
  ```
- **RGtk2**: Interface to Gtk+ for building GUIs.
  ```r
  library(RGtk2)
  window <- gtkWindow("toplevel", show = TRUE)
  window$setTitle("Hello, RGtk2")
  ```

##### **Interactive Reports**
- **flexdashboard**: Easy interactive dashboards for R.
  ```r
  library(flexdashboard)
  flex_dashboard()
  ```
- **pagedown**: Create elegant and interactive web pages using R Markdown.
  ```r
  library(pagedown)
  pagedown::chrome_print("input.Rmd")
  ```

##### **Physics Simulations**
- **rgl**: 3D visualization using OpenGL.
  ```r
  library(rgl)
  plot3d(mtcars$wt, mtcars$mpg, mtcars$hp, col = "red", size = 3)
  ```
- **Physics**: Provides basic physics functionality.
  ```r
  library(Physics)
  setPhysics()
  ```

##### **GIS and Mapping**
- **tmap**: Thematic maps.
  ```r
  library(tmap)
  tm_shape(shp) + tm_polygons("variable")
  ```
- **maptools**: Tools for reading and handling spatial objects.
  ```r
  library(maptools)
  shp <- readShapeSpatial("path/to/shapefile.shp")
  ```

##### **Meta-Analysis**
- **meta**: Meta-Analysis.
  ```r
  library(meta)
  meta_analysis <- metagen(TE = effect, seTE = se, data = df)
  ```
- **metafor**: Meta-Analysis Package for R.
  ```r
  library(metafor)
  res <- rma(yi, vi, data = df)
  ```

##### **3D Graphics**
- **rgl**: 3D visualization using OpenGL.
  ```r
  library(rgl)
  plot3d(x, y, z, col = "blue", size = 3)
  ```
- **plot3D**: Plotting Multi-Dimensional Data.
  ```r
  library(plot3D)
  scatter3D(x, y, z, col = "red")
  ```

##### **Chemistry**
- **chemmineR**: Cheminformatics Toolkit for R.
  ```r
  library(chemmineR)
  sdfset <- read.SDFset("path/to/file.sdf")
  ```
- **rcdk**: Interface to the Chemistry Development Kit.
  ```r
  library(rcdk)
  mol <- parse.smiles("CCO")
  ```

---

### 📈 Basic Data Manipulation
R provides powerful functions for data manipulation:

##### Filtering and Summarizing Data
Using the `dplyr` package from the tidyverse:
```r
# Load dplyr
library(dplyr)

# Filter data
filtered_df <- filter(df, Age > 24)

# Summarize data
summary_df <- df %>%
  group_by(Name) %>%
  summarize(Average_Age = mean(Age))
```

### 📊 Basic Data Visualization
Using the `ggplot2` package from the tidyverse:
```r
# Load ggplot2
library(ggplot2)

# Basic scatter plot
ggplot(df, aes(x = Name, y = Age)) +
  geom_point() +
  labs(title = "Age of Individuals", x = "Name", y = "Age")
```

### 🛠️ Advanced Topics
Once you're comfortable with the basics, explore more advanced topics like:
- **Data cleaning and transformation** using `tidyverse`.
- **Statistical modeling** and **machine learning**.
- **Creating interactive visualizations** with `shiny`.

### 📚 Resources
- **Books**: "R for Data Science" by Hadley Wickham & Garrett Grolemund.
- **Online Courses**: DataCamp, Coursera.
- **Communities**: Stack Overflow, RStudio Community.

#### 📚 Extensive List of R Libraries and Their Uses (Continued)

##### **Econometrics and Time Series Analysis**
- **AER**: Applied Econometrics with R.
  ```r
  library(AER)
  data("CPS1985")
  summary(lm(wage ~ education + experience, data = CPS1985))
  ```
- **tsibble**: Tidy Temporal Data Frames and Tools.
  ```r
  library(tsibble)
  ts <- as_tsibble(df, index = Date)
  ```
- **tseries**: Time series analysis and computational finance.
  ```r
  library(tseries)
  fit <- garch(diff(log(df)), order = c(1, 1))
  ```

##### **Generalized Additive Models**
- **mgcv**: Mixed GAM Computation Vehicle with GCV/AIC/REML smoothness estimation.
  ```r
  library(mgcv)
  fit <- gam(y ~ s(x1) + s(x2), data = df)
  summary(fit)
  ```

##### **Robust Statistical Methods**
- **robustbase**: Basic Robust Statistics.
  ```r
  library(robustbase)
  fit <- lmrob(y ~ x1 + x2, data = df)
  ```

##### **Social Network Analysis**
- **statnet**: Software tools for the representation, visualization, analysis, and simulation of network data.
  ```r
  library(statnet)
  net <- network(matrix)
  plot(net)
  ```
- **sna**: Tools for Social Network Analysis.
  ```r
  library(sna)
  gplot(net)
  ```

##### **Bioinformatics and Genomics**
- **DESeq2**: Differential gene expression analysis based on the negative binomial distribution.
  ```r
  library(DESeq2)
  dds <- DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ condition)
  dds <- DESeq(dds)
  results(dds)
  ```

##### **Chemical Informatics**
- **ChemmineR**: Cheminformatics Toolkit for R.
  ```r
  library(ChemmineR)
  sdfset <- read.SDFset("compounds.sdf")
  ```

##### **Medical Imaging**
- **oro.nifti**: R and Octave interface to the Neuroimaging Informatics Technology Initiative (NIfTI) binary file format.
  ```r
  library(oro.nifti)
  nifti_image <- readNIfTI("path/to/image.nii")
  ```

##### **Climate and Weather Data**
- **rnoaa**: 'NOAA' Weather Data from R.
  ```r
  library(rnoaa)
  weather_data <- ncdc(datasetid = 'GHCND', locationid = 'CITY:US390029', startdate = '2020-01-01', enddate = '2020-12-31')
  ```

##### **Linguistics**
- **qdap**: Quantitative Discourse Analysis Package.
  ```r
  library(qdap)
  q_dap_data <- qdap::transcripts(DATA)
  ```

##### **Image Processing**
- **magick**: Advanced Graphics and Image-Processing in R.
  ```r
  library(magick)
  img <- image_read("path/to/image.jpg")
  image_rotate(img, 90)
  ```

##### **Mathematics and Probability**
- **MCMCpack**: Markov Chain Monte Carlo (MCMC) Package.
  ```r
  library(MCMCpack)
  posterior <- MCMCregress(y ~ x1 + x2, data = df)
  ```
- **actuar**: Actuarial functions and heavy-tailed distributions.
  ```r
  library(actuar)
  dpareto(x, shape = 2, scale = 3)
  ```

##### **Spatial Analysis and Mapping**
- **leaflet**: Create Interactive Web Maps with the JavaScript 'Leaflet' Library.
  ```r
  library(leaflet)
  leaflet(data = quakes) %>%
    addTiles() %>%
    addMarkers(~long, ~lat, popup = ~as.character(mag), label = ~as.character(mag))
  ```
- **spdep**: Spatial Dependence: Weighting Schemes, Statistics.
  ```r
  library(spdep)
  nb <- poly2nb(shp)
  ```

##### **Education and E-Learning**
- **swirl**: Learn R, in R.
  ```r
  library(swirl)
  swirl()
  ```

##### **Financial Market Analysis**
- **blotter**: Transaction Infrastructure for Portfolio Management.
  ```r
  library(blotter)
  portfolio <- initPortf(name = "portfolio", symbols = symbols)
  ```
- **quantstrat**: Quantitative Financial Strategy Speciﬁcation Framework.
  ```r
  library(quantstrat)
  initStrategy(name = "strategy", portfolio = "portfolio")
  ```

##### **Physics and Engineering**
- **pracma**: Practical Numerical Math Functions.
  ```r
  library(pracma)
  quad <- pracma::quadgk(function(x) x^2, 0, 1)
  ```

##### **Survey Analysis**
- **survey**: Analysis of Complex Survey Samples.
  ```r
  library(survey)
  dstrat <- svydesign(id = ~1, strata = ~strata, data = apistrat, fpc = ~fpc)
  ```

##### **Reinforcement Learning**
- **ReinforcementLearning**: Simple Reinforcement Learning Algorithms.
  ```r
  library(ReinforcementLearning)
  env <- gridworldEnvironment
  agent <- ReinforcementLearning::train(env)
  ```

##### **Documentation and Reporting**
- **bookdown**: Authoring Books and Technical Documents with R Markdown.
  ```r
  library(bookdown)
  bookdown::render_book("index.Rmd", "bookdown::gitbook")
  ```
- **officer**: Manipulation of Microsoft Word and PowerPoint Documents.
  ```r
  library(officer)
  doc <- read_docx()
  doc <- body_add_par(doc, "Hello, World!")
  print(doc, target = "output.docx")
  ```

##### **Optimization and Mathematical Programming**
- **ROI**: R Optimization Infrastructure.
  ```r
  library(ROI)
  obj <- L_objective(c(1, 2))
  opt <- OP(obj, bounds = V_bound(1:2, -Inf, Inf))
  result <- ROI_solve(opt)
  ```

##### **Data Import and Export**
- **rio**: A Swiss-Army Knife for Data I/O.
  ```r
  library(rio)
  df <- import("file.xlsx")
  export(df, "file.csv")
  ```

---

### 📈 Basic Data Manipulation
R provides powerful functions for data manipulation:

##### Filtering and Summarizing Data
Using the `dplyr` package from the tidyverse:
```r
# Load dplyr
library(dplyr)

# Filter data
filtered_df <- filter(df, Age > 24)

# Summarize data
summary_df <- df %>%
  group_by(Name) %>%
  summarize(Average_Age = mean(Age))
```

### 📊 Basic Data Visualization
Using the `ggplot2` package from the tidyverse:
```r
# Load ggplot2
library(ggplot2)

# Basic scatter plot
ggplot(df, aes(x = Name, y = Age)) +
  geom_point() +
  labs(title = "Age of Individuals", x = "Name", y = "Age")
```

### 🛠️ Advanced Topics
Once you're comfortable with the basics, explore more advanced topics like:
- **Data cleaning and transformation** using `tidyverse`.
- **Statistical modeling** and **machine learning**.
- **Creating interactive visualizations** with `shiny`.

### 📚 Resources
- **Books**: "R for Data Science" by Hadley Wickham & Garrett Grolemund.
- **Online Courses**: DataCamp, Coursera.
- **Communities**: Stack Overflow, RStudio Community.

Enjoy your journey with R! Happy coding! 🎉
