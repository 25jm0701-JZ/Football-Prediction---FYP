import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const [, , csvPath, xlsxPath] = process.argv;
if (!csvPath || !xlsxPath) {
  throw new Error(
    "Usage: node export_poisson_predictions_xlsx.mjs input.csv output.xlsx",
  );
}

const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Predictions" });
const sheet = workbook.worksheets.getItem("Predictions");
const rowCount = csvText.trimEnd().split(/\r?\n/).length;
const bodyEnd = Math.max(rowCount, 2);

sheet.showGridlines = false;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(3);

const fullRange = sheet.getRange(`A1:X${bodyEnd}`);
fullRange.format = {
  font: { name: "Aptos", size: 10, color: "#1F2937" },
  verticalAlignment: "center",
  borders: { preset: "inside", style: "hair", color: "#E5E7EB" },
};

const header = sheet.getRange("A1:X1");
header.format = {
  fill: "#1F4E78",
  font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D6E4F0" },
  rowHeight: 32,
};

sheet.getRange(`A2:A${bodyEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`B2:B${bodyEnd}`).format.numberFormat = "yyyy-mm-dd hh:mm";
sheet.getRange(`C2:C${bodyEnd}`).format.numberFormat = "@";
sheet.getRange(`D2:E${bodyEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`H2:I${bodyEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`L2:X${bodyEnd}`).format.horizontalAlignment = "center";

sheet.getRange(`L2:M${bodyEnd}`).format.numberFormat = "0.000";
sheet.getRange(`N2:P${bodyEnd}`).format.numberFormat = "0.0%";
sheet.getRange(`R2:R${bodyEnd}`).format.numberFormat = "0.00%";
sheet.getRange(`T2:T${bodyEnd}`).format.numberFormat = "0.00%";
sheet.getRange(`V2:V${bodyEnd}`).format.numberFormat = "0.00%";
sheet.getRange(`X2:X${bodyEnd}`).format.numberFormat = "0.00%";

for (const column of ["Q", "S", "U", "W"]) {
  sheet.getRange(`${column}2:${column}${bodyEnd}`).format.numberFormat = "@";
}

const widths = {
  A: 12,
  B: 22,
  C: 19,
  D: 11,
  E: 12,
  F: 23,
  G: 23,
  H: 11,
  I: 15,
  J: 28,
  K: 22,
  L: 14,
  M: 14,
  N: 13,
  O: 13,
  P: 13,
  Q: 16,
  R: 18,
  S: 11,
  T: 11,
  U: 11,
  V: 11,
  W: 11,
  X: 11,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}

for (let row = 2; row <= bodyEnd; row += 2) {
  sheet.getRange(`A${row}:X${row}`).format.fill = "#F5F9FC";
}

await fs.mkdir(path.dirname(xlsxPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
