temp = ispan(1, 100, 1)

satvpr = satvpr_temp_fao56(temp, (/2,2/))

system("/bin/rm -f satvpr_temp_fao56_output.nc")   ; remove any pre-existing file
ncdf = addfile("satvpr_temp_fao56_output.nc" ,"c")  ; open output netCDF file
filedimdef(ncdf,"time",-1,True)

ncdf->satvpr = satvpr
