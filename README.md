# Prognozowanie aktywności wrzodziejącego zapalenia jelita grubego na podstawie
biomarkerów z wykorzystaniem metod uczenia maszynowego
# Spis treści
* [Opis projektu](#opis-projektu)
* [Technologia](#technologia)

## Opis projektu
Celem projektu było zaprojektowanie modelu uczenia maszynowego na podstawie rutynowo wykonywanych laboratoryjnych badań krwi, moczu i kału w celu wsparcia różnicowania między poziomami aktywności wrzodziejącego zapalenia jelita grubego. 
Kolejnym celem było uzyskanie jak najwyższego poziomu dokładności algorytmu, a także określenie wpływu biomarkerów na wynik klasyfikacji modelu.

Porównano wyniki dokładności prognozowania następujących klasyfikatorów
* drzewa decyzyjne
* metoda k-najbliższych sąsiadów
* maszyny wektorów nośnych (SVM – support vector machine)
* naiwny klasyfikator Bayesa
* las losowy
* regresja logistyczna
* dodatkowe drzewa
	
Dane wejściowe
* dane pochodzą od pacjentów (wybrano 56 najbardziej istotnych biomarkerów)
* wykorzystano metodę regresji do uzupełnienia brakujących wartości w danych
* wykorzystano technikę SMOTE aby zlikwidować problem nierównowagi klas
 * Klasa 0: 45 rekordów danych
 * Klasa 1: 64 rekordy danych
 * Klasa 2: 88 rekordów danych
 * Klasa 3: 55 rekordów danych
## Technologia
Projekt został utworzony przy użyciu:
* Python 3.10.9
