# Design codes and standards

Local copies of the documents the structure rules (`src/structure_decide.py`, `config/box_sizes.csv`) are based on, copied read-only from `\\145.145.51.51\Technical Data\02. H&H Drainage\`. The documents themselves are not in git (see `.gitignore`); this index is.

Paths below are relative to that share folder.

## RDSO_box_standards/

| File | What it gives | Share path |
|---|---|---|
| `Box culvert details-RDSO.ods` | **Source of `config/box_sizes.csv`.** "DFC Loading" tables: single cell double track (38 sizes, 1x1.2 to 6x6), single cell single track (0-10 m fill), twin cell single track (4-6 m spans, intermediate wall 0.4-0.5 m) | `Codes & Standards\India\IR\Standard Culverts\` |
| `10152-1R.PDF` | RDSO/B-10152/1R single box, single track, 25t, 0 m fill - spans 2-6 m, heights 1.8-6 m | `Codes & Standards\India\IR\Standard Culverts\` |
| `RDSO-B-10158-1.PDF` | RDSO/B-10158/1 twin box, double line straight, 0 m fill - spans 3-6 m | `Codes & Standards\India\IR\` |
| `10153.1.jpg`, `10153.2.jpg` | RDSO/B-10153 segmental box for Limited Height Subway (4-5.5 m x 3.6-4 m) | `Codes & Standards\India\IR\` |
| `10155-1.jpg` | RDSO/B-10155/1 single box, double line curved up to 4 deg | `Codes & Standards\India\IR\` |
| `1_2X1_2rcc box.jpg` | 1 x 1.2 x 1.2 m precast box drawing | `Codes & Standards\India\IR\Standard Culverts\` |
| `RDSO, 2018-08, Drawing Master List.pdf` | RDSO drawing list - RCC box series incl. DFC loading (pp. 28-29) | `Codes & Standards\India\IR\` |
| `RDSO, Bridges & Structures (BS) Code List.pdf` | RDSO BS guideline list (BS-112 ROB planning, BS-120 LHS, BS-138 RUB - not on the share) | `Codes & Standards\India\IR\` |

## Indian_Railways/

| File | What it gives | Share path |
|---|---|---|
| `2018-09, Railways Board, RCC Box Circular.pdf` | Railway Board 18.09.2018: RCC box may be used for RUB/LHS; avoid multiple and segmental boxes | `Codes & Standards\India\IR\` |
| `IRBM 24.pdf` | Indian Railways Bridge Manual 2024 - para 301(3) restrict skew to 30 deg (minor bridges), 311(3) min span/headroom | `Codes & Standards\India\IR\` |
| `CS_25_BRIDGE MANUAL.pdf` | IRBM correction slip 25 - min 1 m span, 1.2 m headroom | `Codes & Standards\India\IR\Standard Culverts\` |
| `2014, CS 29  IRBM (VC & Freeboard).pdf` | IRBM correction slip 29 - vertical clearance and freeboard | `Codes & Standards\India\IR\IR, Indian Railways Bridge Manual\` |
| `2019, Railway Bridge Planning Final.pdf` | IRICEN Bridge Planning 2019 - §7.4 choice of superstructure (box vs slab/girder) | `Codes & Standards\India\IR\IR, Indian Railways Bridge Manual\` |
| `2010, Classification of Bridges.pdf` | Culvert / minor / major / important bridge definitions | `Codes & Standards\India\IR\` |
| `2012, RDSO Freeboard & Vertical Clearance.pdf` | RDSO freeboard and vertical clearance | `Codes & Standards\India\IR\` |
| `Indian Railway code for engineering dept. - 2012.pdf` | Engineering Code (scanned) - para 1816(iii) road widths through ROB/RUB (2-lane 7.5 m + footpaths, 4-lane NH 24 m incl. median) | `Codes & Standards\India\IR\` |

## IRC_road/

| File | What it gives | Share path |
|---|---|---|
| `SP 84, 2014, 4-Laning of Highways through PPP.pdf` | §2.10 underpass clearances (5.5 m vehicular), §7.1(ii) independent structure per direction, §7.18.2 ROB skew > 45 deg -> realign, Table 2.2 medians | `Codes & Standards\India\IRC\` |
| `SP 73, 2015, 2-Laning of Highways through PPP.pdf` | 2-lane equivalents of the SP-84 clauses | `Codes & Standards\India\IRC\` |
| `IRC-SP-13 -2022-Small Bridges and culverts.pdf` | §14.5 box cells: single up to 8 m, double/triple up to 3 m per cell (road culverts) | `Codes & Standards\India\IRC\` |
| `IRC 5, 2015, Standard Specs for Road Bridges - I. General Features.pdf` | IRC:5 general features of road bridges (scanned) | `Codes & Standards\India\IRC\` |
| `BRIDGE_DESIGN_GUIDELINES_FOR_BRIDGE_DESI.pdf` | Maharashtra PWD bridge design guidelines - RCC slab skew bridges 15-60 deg | `Codes & Standards\India\IRC\` |

## org_templates/

| File | What it gives | Share path |
|---|---|---|
| `Project Code_Bridge List_xx+xxx to xx+xxx km_Rx.xlsx` | Organisation bridge-list template (filled: Hyderabad-Bangalore 4A). ReferenceData: vertical clearance major road 5.5 m / minor road 4.0 m; StdSpanData: formation-to-soffit depths per span | `References\Optimization\Excel Automations With Macro\Reference documents\` |
| `Reference_Calculation_Sheet_Culvert_& _Slab_40-80Km_250903.ods` | Chalisgaon-Aurangabad culvert summary - box/slab sizes used in practice | `References\Optimization\Excel Automations With Macro\Reference documents\` |
| `Culvert_& _Slab_250904_EWDFC.ods` | EWDFC culvert/slab calculation sheet - standard box list, skew span formula (box keeps its square span) | `References\Optimization\Excel Automations With Macro\Excel Sheet Automation\Culverts and Slabs\Single Line Calculations\EWDFC_Sheets\` |
| `EWDFC_Yavalkhed__Rational_Reference sheet.ods` | Skew number convention example | same folder as above |
| `QGIS_Skew_Number_Script_250929_v02.py`, `Skew_Angles.model3` | Organisation skew algorithm: skew number 0-180 clockwise from track, skew angle = \|90 - skew number\| | `References\Optimization\QGIS\Models\` (`Scripts_by_Peter\` for the .py) |

## asset_registers/

| File | What it gives | Share path |
|---|---|---|
| `AssetRegister_BRIDGE - TVC - SRR.xls`, `AssetRegister_BRIDGE SRR - MAQ.xls` | Southern Railway TMS bridge registers (5,141 bridges incl. RUB/LHS/ROB with span and skew) - what box/slab sizes are built in practice | `Project\2548_SR\Section_7\gad\GADs from client\GAD Remaining\GAD Remaining\SECTION 6A\section-6\ROB GAD list 1\Bridge List - TMS data\` |

## checklists/

`CL-RLY-09`, `-110`, `-115` (box culverts), `-119` (at-grade crossings), `-120` (underpass crossings) from `ISO Checklists\` - what a design review checks; no thresholds of their own.

## Not on the share

RDSO BS-138 (RUB/subway), BS-112 (ROB planning), BS-120 (LHS), the DFC-loading RDSO box drawing sheets themselves (B-10151/10154/10156/10157 - only listed in the master list), any DFCCIL road-crossing guideline, IRC:SP:87, IRC:SP:90.
