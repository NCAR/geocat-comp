t1  = (/ 104, 100, 92, 92, 86, 80, 80, 60, 30 /)    ; degF
rh1 = (/  55,  65, 60, 90, 90, 40, 75, 90, 50 /)    ; %
io = (/2,2/)                                       ; degF input, degF output
hi1 = heat_index_nws(t1, rh1, io, False)

t2  = (/ 70, 75, 80, 85, 90, 95, 100, 105, 110, 115 /)    ; degF
rh2 = (/ 10, 75, 15, 80, 65, 25, 30,  40,  50,  5 /)    ; %
hi2 = heat_index_nws(t2, rh2, io, False)

system("/bin/rm -f heat_index_output.nc")   ; remove any pre-existing file
ncdf = addfile("heat_index_output.nc" ,"c")  ; open output netCDF file
filedimdef(ncdf,"time",-1,True)

ncdf->hi1 = hi1
ncdf->hi2 = hi2
