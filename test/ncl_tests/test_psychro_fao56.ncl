temp = ispan(1, 100, 1)

psychro = psychro_fao56(temp, (/2,2/))

system("/bin/rm -f psychro_fao56_output.nc")   ; remove any pre-existing file
ncdf = addfile("psychro_fao56_output.nc" ,"c")  ; open output netCDF file
filedimdef(ncdf,"time",-1,True)

ncdf->psychro = psychro
