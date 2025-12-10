;*************************************************
; generates an output array and saves it
;*************************************************

begin

;*************************************************
; read in data
;*************************************************

in = addfile("atmos.nc","r")
ps = in->PS(0, :, :)
hyam = in->hyam
hybm = in->hybm
p0 = 1000.0*100

;*************************************************
; calculate
;*************************************************

dph = dpres_hybrid_ccm(ps, p0, hyam, hybm)

dph!0 = "lev"
dph!1 = "lat"
dph!2 = "lon"
dph&lat = ps&lat
dph&lon = ps&lon

;*************************************************
; write out to file
;*************************************************

system("/bin/rm -f ./dpres_hybrid_ccm_output.nc")   ; remove any pre-existing file
ncdf = addfile("./dpres_hybrid_ccm_output.nc" ,"c")  ; open output netCDF file

ncdf->dph  = dph

end
