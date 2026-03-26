#!/usr/bin/env Rscript
# Generate harmony2 reference data for harmonypy tests.
#
# Prerequisites:
#   Install harmony from the harmony2 branch:
#     remotes::install_github("immunogenomics/harmony@harmony2")
#
# Usage:
#   cd harmonypy
#   Rscript scripts/generate_harmony2_reference.R

library(harmony)

cat("harmony version:", as.character(packageVersion("harmony")), "\n")

generate_reference <- function(meta_tsv, pcs_tsv, output_tsv, batch_var) {
    cat("\n--- Processing:", meta_tsv, "---\n")

    meta <- read.delim(meta_tsv)
    pcs <- read.delim(pcs_tsv, check.names = FALSE)

    # Drop row names column if present
    if (is.character(pcs[[1]])) {
        rownames(pcs) <- pcs[[1]]
        pcs <- pcs[, -1]
    }

    cat("Cells:", nrow(pcs), " PCs:", ncol(pcs), "\n")
    cat("Batch variable:", batch_var, "\n")
    cat("Batch levels:", paste(unique(meta[[batch_var]]), collapse = ", "), "\n")

    # RunHarmony expects cells in rows; it will transpose internally
    result <- RunHarmony(
        as.matrix(pcs),
        meta,
        batch_var,
        verbose = TRUE,
        return_object = FALSE
    )

    cat("Result shape:", nrow(result), "x", ncol(result), "\n")

    write.table(
        result,
        gzfile(output_tsv),
        sep = "\t",
        quote = FALSE,
        row.names = TRUE
    )
    cat("Saved:", output_tsv, "\n")
}

# Small dataset
generate_reference(
    meta_tsv = "data/pbmc_3500_meta.tsv.gz",
    pcs_tsv = "data/pbmc_3500_pcs.tsv.gz",
    output_tsv = "data/pbmc_3500_pcs_harmony2.tsv.gz",
    batch_var = "donor"
)

# Medium dataset
if (file.exists("data/ircolitis_blood_cd8_obs.tsv.gz")) {
    generate_reference(
        meta_tsv = "data/ircolitis_blood_cd8_obs.tsv.gz",
        pcs_tsv = "data/ircolitis_blood_cd8_pcs.tsv.gz",
        output_tsv = "data/ircolitis_blood_cd8_pcs_harmony2.tsv.gz",
        batch_var = "batch"
    )
}

# Large dataset
if (file.exists("data/acute_myeloid_obs.tsv.gz")) {
    generate_reference(
        meta_tsv = "data/acute_myeloid_obs.tsv.gz",
        pcs_tsv = "data/acute_myeloid_pcs.tsv.gz",
        output_tsv = "data/acute_myeloid_pcs_harmony2.tsv.gz",
        batch_var = "batch"
    )
}

cat("\nDone! Reference data generated.\n")
