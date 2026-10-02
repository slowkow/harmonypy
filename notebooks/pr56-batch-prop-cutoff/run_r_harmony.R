#!/usr/bin/env Rscript
# Run R harmony on the same toy and ircolitis inputs as run_harmonypy.py.
#
# Needs harmony from the harmony2 branch built with LAPACK: the two-column
# ridge step calls inv(). See run_all.sh for a macOS build that links
# Accelerate. Usage (from this folder):
#   Rscript run_r_harmony.R
suppressPackageStartupMessages({
    library(harmony)
    library(data.table)
})
cat("harmony", as.character(packageVersion("harmony")),
    "from", find.package("harmony"), "\n")

dir.create("results", showWarnings = FALSE)
label <- "r_harmony"

lab_gap <- function(xy, toy, type) {
    in_type <- toy$cell_type == type
    a <- colMeans(xy[in_type & toy$lab == "A", , drop = FALSE])
    b <- colMeans(xy[in_type & toy$lab == "B", , drop = FALSE])
    sqrt(sum((a - b)^2))
}

# Toy sweep ---------------------------------------------------------------
toy <- fread("data/toy.tsv", colClasses = list(character = c("lab", "day")))
xy <- as.matrix(toy[, .(x, y)])
cutoffs <- round(seq(0, 0.3, by = 0.005), 3)
rows <- list()
for (cutoff in cutoffs) {
    set.seed(1)
    out <- RunHarmony(
        xy, toy, c("lab", "day"), theta = c(0, 0), lambda = c(1, 1),
        nclust = 2, verbose = FALSE,
        .options = harmony_options(batch.prop.cutoff = cutoff)
    )
    rows[[length(rows) + 1]] <- data.table(
        cutoff = cutoff,
        gap_type1 = lab_gap(out, toy, 1),
        gap_type2 = lab_gap(out, toy, 2)
    )
    if (isTRUE(all.equal(cutoff, 0.15))) {
        fwrite(data.table(x = out[, 1], y = out[, 2]),
               sprintf("results/toy_corrected_%s.tsv", label), sep = "\t")
    }
}
fwrite(rbindlist(rows), sprintf("results/toy_sweep_%s.tsv", label), sep = "\t")

# ircolitis blood CD8, corrected for donor and batch ----------------------
obs <- fread("../../data/ircolitis_blood_cd8_obs.tsv.gz",
             select = c("donor", "batch"))
pcs <- fread("../../data/ircolitis_blood_cd8_pcs.tsv.gz")
pcs <- as.matrix(pcs[, grep("^PC\\d+$", names(pcs), value = TRUE), with = FALSE])
for (cutoff in c(1e-5, 1e-2)) {
    set.seed(1)
    out <- RunHarmony(
        pcs, obs, c("donor", "batch"), verbose = FALSE,
        .options = harmony_options(batch.prop.cutoff = cutoff)
    )
    fwrite(as.data.table(out),
           sprintf("results/ircolitis_%s_cutoff%g.tsv.gz", label, cutoff),
           sep = "\t")
}
