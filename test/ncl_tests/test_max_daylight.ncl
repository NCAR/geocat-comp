lat     = ispan(-66, 66, 1)
time    = ispan(1, 365, 1)

sunmax = daylight_fao56(time, lat)

system("/bin/rm -f max_daylight_test.nc")   ; remove any pre-existing file
ncdf = addfile("max_daylight_test.nc" ,"c")  ; open output netCDF file
filedimdef(ncdf,"time",-1,True)

ncdf->sunmax = sunmax