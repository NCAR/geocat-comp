;*************************************************
; grid_fill_3.ncl
;*************************************************
;
; Concepts illustrated:
;   - Reading model-generated data on hybrid levels
;   - Using vinth2p to interpolate to constant pressure levels
;   - Setting parameters for "poisson_grid_fill"
;   - Using "poisson_grid_fill" to fill grid locations
;
;*************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin
; diri = "/Users/shea/Data/CAM/"
 diri = "./"
 fili = "ccsm35.h0.0021-01.demo.nc"

 f    = addfile (diri+fili, "r")

 lev_p = (/100,200,300,400,500, 600, 700, 750, 850, 925, 950,1000/)

 lev_p!0         = "lev_p"                  ; variable/dim name
 lev_p&lev_p     =  lev_p                   ; create coordinate variable
 lev_p@long_name = "pressure"               ; attach some attributes
 lev_p@units     = "hPa"
 lev_p@positive  = "down"

 hyam = f->hyam
 hybm = f->hybm
 P0mb = 1000.                        ; reference pressure [mb]
 PS   = f->PS
 T    = f->T                         ; MODEL on hybrid levels
 Q    = f->Q
 TS   = f->TS
 Z    = f->Z3
 PHIS = f->PHIS

 Q    = Q*1000.                      ; make g/kg for nicer plot
 Q@units = "g/kg"

;-----------------------------------------------------------------
; Interpolate and extrapolation below surface pressure
;-----------------------------------------------------------------

 varflg = 1
 Tpx    = vinth2p_ecmwf (T,hyam,hybm,lev_p,PS,1,P0mb,1,True,varflg,TS,PHIS)
 varflg = 0
 Qpx    = vinth2p_ecmwf (Q,hyam,hybm,lev_p,PS,1,P0mb,1,True,varflg,TS,PHIS)
 varflg = -1
 Zpx    = vinth2p_ecmwf (Z,hyam,hybm,lev_p,PS,1,P0mb,1,True,varflg,TS,PHIS)

;===========================
; write output to netcdf
;===========================

   system("/bin/rm -f vinth2p_ecmwf_output.nc")   ; remove any pre-existing file
   ncdf = addfile("vinth2p_ecmwf_output.nc" ,"c")  ; open output netCDF file

;===================================================================
; create global attributes of the file (optional)
;===================================================================
   fAtt               = True            ; assign file attributes
   fAtt@title         = "vinth2p_ecmwf output of grid_fill_3 example"
   fAtt@source_file   =  "original-file.nc"
   fAtt@Conventions   = "None"
   fAtt@creation_date = systemfunc ("date")
   fileattdef( ncdf, fAtt )            ; copy file attributes

;===================================================================
; make time an UNLIMITED dimension; recommended  for most applications
;===================================================================
   filedimdef(ncdf,"time",-1,True)

;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================

   ncdf->T  = Tpx
   ncdf->Q  = Qpx
   ncdf->Z  = Zpx

end