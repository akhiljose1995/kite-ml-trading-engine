import pandas as pd

class ADXSignalDetector:
    """
    Detects trading signals based on ADX, +DI, and -DI combinations.
    """

    def __init__(self, df, adx_periods=[14]):
        """
        Initialize with a DataFrame containing ADX, +DI, -DI data.

        :param df: DataFrame with columns 'ADX_{p}', '+DI_{p}', '-DI_{p}' for each period in adx_periods
        :param adx_periods: List of periods used for ADX and DI calculation
        """
        self.df = df.copy()
        self.adx_periods = adx_periods
        self.window = 3
    
    def _is_strictly_increasing(self, series):
        return all(series[-1] > val for val in series[:-1])

    def _is_strictly_decreasing(self, series):
        return all(series[-1] < val for val in series[:-1])

    def detect_signals(self):
        """
        Detects signals based on combinations of ADX and DI indicators.
        Below are scenarios:

        ADX (Trend Strength)
        Case	Description	Meaning
        A1	ADX < 25	Weak/No trend — avoid
        A2	ADX < 25 and rising	Trend developing — early entry zone
        A3	ADX > 25 and rising	Strong trend strengthening — ideal zone
        A4	ADX > 25 and falling toward 25	Trend weakening — caution zone
        A5	ADX > 35	Very strong trend — possible exhaustion approaching

        +DI
        Case	Description
        D1	+DI > -DI and falling
        D2	+DI > -DI and rising

        -DI
        Case	Description
        D3	-DI > +DI and falling
        D4	-DI > +DI and rising

        DI Crossovers
        Case	Description
        C1	+DI above -DI but both converging
        C2	-DI above +DI but both converging

        Consider below combinations 
        ADX Case	DI Case	Signal
        A3 + D2	Strong Uptrend	"BUY"
        A3 + D4	Strong Downtrend	"SELL"
        A2 + D2	Early Uptrend	"EARLY BUY"
        A2 + D4	Early Downtrend	"EARLY SELL"
        A3 + C1	+DI & -DI converge	"SELL"
        A3 + C2	-DI & +DI converge	"BUY"
        A1	Any	"NO TRADE"
        A4 + any cross	"EXIT"	
        A5 + +DI or -DI weakening	"EXIT"
        :return: Dictionary of {period: signal_series}
        """
        signals = {}

        for period in self.adx_periods:
            adx_col = f'ADX_{period}'
            pdi_col = f'+DI_{period}'
            ndi_col = f'-DI_{period}'

            adx = self.df[adx_col]
            pdi = self.df[pdi_col]
            ndi = self.df[ndi_col]

            signal = None
            
            adx_vals = adx.iloc[-self.window:-1].values
            pdi_vals = pdi.iloc[-self.window:-1].values
            ndi_vals = ndi.iloc[-self.window:-1].values

            latest_adx = adx_vals[-1]
            latest_pdi = pdi_vals[-1]
            latest_ndi = ndi_vals[-1]

            # Direction detection
            adx_rising = self._is_strictly_increasing(adx_vals)
            adx_falling = self._is_strictly_decreasing(adx_vals)
            pdi_rising = self._is_strictly_increasing(pdi_vals)
            pdi_falling = self._is_strictly_decreasing(pdi_vals)
            ndi_rising = self._is_strictly_increasing(ndi_vals)
            ndi_falling = self._is_strictly_decreasing(ndi_vals)

            # Determine ADX Case
            a_case = None
            if latest_adx < 25:
                a_case = "A2" if adx_rising else "A1"
            elif latest_adx > 35:
                if (pdi_falling and latest_pdi > latest_ndi) or \
                    (ndi_falling and latest_ndi > latest_pdi):
                    a_case = "A5"
                else:
                    a_case = "A3"
            elif latest_adx > 25:
                a_case = "A3" if adx_rising else "A4"

            # DI Case
            d_case = None
            if latest_pdi > latest_ndi:
                d_case = "D2" if pdi_rising else "D1"
            elif latest_ndi > latest_pdi:
                d_case = "D4" if ndi_rising else "D3"

            # DI Crossovers (Convergence)
            c_case = None
            spread_now = abs(latest_pdi - latest_ndi)
            spread_past = abs(pdi_vals[0] - ndi_vals[0])
            if latest_pdi > latest_ndi and spread_now < spread_past:
                c_case = "C1"
            elif latest_ndi > latest_pdi and spread_now < spread_past:
                c_case = "C2"

            # Signal Mapping
            final_signal = None
            if a_case == "A1":
                final_signal = "NO TRADE"
            elif a_case == "A2":
                if d_case == "D2":
                    final_signal = "EARLY BUY"
                elif d_case == "D4":
                    final_signal = "EARLY SELL"
            elif a_case == "A3":
                if d_case == "D2":
                    final_signal = "BUY"
                elif d_case == "D4":
                    final_signal = "SELL"
                elif c_case == "C1":
                    final_signal = "SELL"
                elif c_case == "C2":
                    final_signal = "BUY"
            elif a_case == "A4" and c_case:
                final_signal = "EXIT"
            elif a_case == "A5" and ((pdi_falling and latest_pdi > latest_ndi) or
                                        (ndi_falling and latest_ndi > latest_pdi)):
                final_signal = "EXIT"

            ## Add the final signal to the DataFrame, if needed.
            #self.df[f'signal_{period}'] = final_signal
            signals[period] = final_signal

        return signals
