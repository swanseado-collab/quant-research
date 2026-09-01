from numba import njit
import strategy_grid as s

# strategy_grid.simulate is lazily compiled. Bind the helper as a Numba dispatcher
# before the first simulate() call so nopython mode can resolve it.
s.update_avg = njit(cache=True)(s.update_avg)

if __name__ == '__main__':
    s.main()
