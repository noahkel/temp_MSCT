# Spektraldaten für P1

Abgerufen am 17.11.2026. Alle zum Ausführen benötigten Zahlen liegen lokal vor;
das Programm benötigt keine Internetverbindung.

## Massenschwächung

`nist_water_bone.csv` enthält die NIST-Stützstellen von 20 bis 150 keV:

- Wasser: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
- Kortikaler Knochen (ICRU-44): https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html
- Referenzdichten: https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

Übernommen wurde die Spalte **mu/rho**, nicht **mu_en/rho**.
Energie wurde von MeV in keV umgerechnet (Faktor 1000).
Die Koeffizienten bleiben in cm²/g. `attenuation` interpoliert log-log.
Für diesen eingeschränkten Energiebereich und Wasser/Knochen werden keine
K-Kanten übersprungen. Diese Interpolation darf nicht unverändert auf Jod
übertragen werden. Referenzdichten für unser Phantom: 1.00 und 1.92 g/cm³.
Die Dichten stehen in f; sie werden nicht nochmals in M hineingerechnet.

## Quellspektrum

`source_spectrum.csv` ist eine CSV-Konvertierung von
`Spectral/ScannerModel/incidentSpectrum.mat`, Variable `incidentSpectrum`, aus:

https://github.com/SimonRit/OneStepSpectralCT

Verwendeter Commit: `3e3102e41b0d583736d1b62389e844767c6a0466`.
Dateilink:
https://github.com/SimonRit/OneStepSpectralCT/blob/3e3102e41b0d583736d1b62389e844767c6a0466/Spectral/ScannerModel/incidentSpectrum.mat

Das Repository gehört zur Arbeit von Mory et al. (2018), *Comparison of five
one-step reconstruction algorithms for spectral CT*. Prohaszka et al. (2024),
Abschnitt 4.2, verwenden dieses Referenzprojekt ebenfalls.
`SimulatePreps.m` bezeichnet die Daten als Philips-simuliertes Spektrum.
Es handelt sich nicht um eine eigene Messung und nicht um ein hier neu
kalibriertes 120-kV-Spektrum. Röhrenparameter werden daher nicht behauptet.

Die 150 Werte werden unverändert übernommen, mit Energien 1,...,150 keV.
Diese Indexkonvention folgt den energieindizierten Tabellen und Bin-Indizes
im Referenzcode (`SimulatePreps.m`, `ExportPhilipsSpectralData.m`).
P1 verwendet davon 20,...,120 keV und normiert anschließend jedes Bin.
Die absolute Photonenzahl ist damit irrelevant; I0 wird als relative
spektrale Gewichtung verwendet. dE=1 keV ist in S enthalten.

Die im Referenzprojekt ebenfalls vorhandene nichtideale Detektorantwort
wird **nicht** übernommen: Hier gelten die ideal disjunkten Bins aus dem
Arbeitsplan. Somit ist P1 keine Reproduktion des dortigen Gesamtversuchs.

Das Referenzrepository stellt seine Inhalte unter Apache-2.0 bereit.
Die Lizenz liegt unverändert als `LICENSE_OneStepSpectralCT.txt` bei.
Änderung an den übernommenen Spektraldaten: ausschließlich Dateiformat
MATLAB -> CSV und explizite Energiespalte; keine Änderung der Intensitätswerte.
