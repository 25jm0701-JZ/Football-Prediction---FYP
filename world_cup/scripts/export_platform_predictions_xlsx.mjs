import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const [, , csvPath, xlsxPath] = process.argv;
if (!csvPath || !xlsxPath) {
  throw new Error(
    "Usage: node export_platform_predictions_xlsx.mjs input.csv output.xlsx",
  );
}

const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, {
  sheetName: "Platform Predictions",
});
const sheet = workbook.worksheets.getItem("Platform Predictions");
const rowCount = csvText.trimEnd().split(/\r?\n/).length;
const bodyEnd = Math.max(rowCount, 2);

sheet.showGridlines = false;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(7);

const fullRange = sheet.getRange(`A1:AV${bodyEnd}`);
fullRange.format = {
  font: { name: "Aptos", size: 10, color: "#1F2937" },
  verticalAlignment: "center",
  borders: { preset: "inside", style: "hair", color: "#E5E7EB" },
};

const headerBlocks = [
  ["A1:K1", "#355C7D"],
  ["L1:S1", "#1F4E78"],
  ["T1:AC1", "#2E7D6E"],
  ["AD1:AG1", "#B7791F"],
  ["AH1:AP1", "#6B5B95"],
  ["AQ1:AV1", "#7A3E65"],
];
for (const [range, fill] of headerBlocks) {
  sheet.getRange(range).format = {
    fill,
    font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D9E2F3" },
    rowHeight: 38,
  };
}

sheet.getRange(`A2:E${bodyEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`H2:I${bodyEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`M2:AV${bodyEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`B2:B${bodyEnd}`).format.numberFormat = "yyyy-mm-dd hh:mm";
sheet.getRange(`C2:C${bodyEnd}`).format.numberFormat = "@";

for (const range of [
  `N2:P${bodyEnd}`,
  `W2:W${bodyEnd}`,
  `Y2:Y${bodyEnd}`,
  `AA2:AA${bodyEnd}`,
  `AC2:AC${bodyEnd}`,
  `AH2:AP${bodyEnd}`,
  `AQ2:AS${bodyEnd}`,
  `AT2:AV${bodyEnd}`,
]) {
  sheet.getRange(range).format.numberFormat = "0.0%";
}
sheet.getRange(`Q2:S${bodyEnd}`).format.numberFormat = "0.000";
sheet.getRange(`T2:U${bodyEnd}`).format.numberFormat = "0.000";
sheet.getRange(`AE2:AE${bodyEnd}`).format.numberFormat = "0.0%";

for (const column of ["V", "X", "Z", "AB"]) {
  sheet.getRange(`${column}2:${column}${bodyEnd}`).format.numberFormat = "@";
}

const widths = {
  A: 11, B: 18, C: 18, D: 10, E: 11, F: 22, G: 22, H: 10, I: 13,
  J: 27, K: 20, L: 24, M: 14, N: 12, O: 12, P: 12, Q: 15, R: 15,
  S: 15, T: 13, U: 13, V: 16, W: 17, X: 10, Y: 10, Z: 10, AA: 10,
  AB: 10, AC: 10, AD: 15, AE: 17, AF: 17, AG: 15, AH: 11, AI: 11,
  AJ: 11, AK: 14, AL: 14, AM: 14, AN: 11, AO: 11, AP: 11, AQ: 15,
  AR: 15, AS: 15, AT: 14, AU: 14, AV: 14,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}

for (let row = 2; row <= bodyEnd; row += 2) {
  sheet.getRange(`A${row}:AV${row}`).format.fill = "#F6F8FA";
}

await fs.mkdir(path.dirname(xlsxPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
