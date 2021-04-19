freq = 10000

t = np.arange(30000) / freq  # ten seconds of 1kHz

t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
          np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
          np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
          np.sin(100 * t * m.tau) / 10)

s_data_lp = fourier_low_pass(t_data, freq, 15)
s_data_hp = fourier_high_pass(t_data, freq, 15)
s_data_bp = fourier_band_pass(t_data, freq, 3, 30)
s_data_bb = fourier_band_block(t_data, freq, 3, 30)
res = freq / len(t)
x_axis = np.array([x * res for x in range(len(t))])

#plt.semilogx(x_axis[:400], np.fft.fft(s_data_lp)[:400])
plt.plot(t_data)
# plt.plot(flp(t_data, freq, 30))
plt.plot(s_data_bb)
plt.show()
