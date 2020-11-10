library(swmmr)
library(purrr)
library(dplyr)
library(sf)

# inp_file <- system.file("C:/PycharmProjects/swmm_rl_hague/hague_inp_files/hague_v11_template_WQ.inp")
# inp_path <- system.file("extdata", "Example1.inp", package = "swmmr", mustWork = TRUE)
inp_path <- read_inp(x = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files/hague_v11_template_WQ.inp")
inp <- read_inp(x = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files/hague_v11_template_WQ_nosubs.inp")

# convert .inp file into independent .shp and .txt files
inp_to_files(x = inp, name = "Hague", path_out = "C:/PycharmProjects/swmm_rl_hague/hague2arc_swmmr")
