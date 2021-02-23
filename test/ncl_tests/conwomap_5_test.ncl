;***********************
; conwomap_5.ncl
;
; Concepts illustrated:
;   - Drawing a simple contour plot
;   - Making an axis logarithmic in a contour plot
;   - Changing the labels and tickmarks on a contour plot
;   - Creating a main title
;   - Attaching coordinate arrays to a variable
;
;***********************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;===========================
begin
;*************************
; get data
;*************************
 f     = addfile("atmos.nc","r")
 u     = f->U(0,:,:,:)
;*************************
; convert to pressure levels
;*************************
  hyam   = f->hyam
  hybm   = f->hybm
  ps     = f->PS
  p0     = 1000.
  pres3d      = (/1000,950,800,700,600,500,400,300,200/)
  pres3d@units= "mb"

  u_int=(/vinth2p(u,hyam,hybm,pres3d,ps(0,:,:),2,\
              p0,2,False)/)
  u_int!0    = "plev"
  u_int&plev = pres3d
  u_int!1    = "lat"
  u_int&lat  = u&lat
  u_int!2    = "lon"
  u_int&lon  = u&lon
  u_int@long_name = "Zonal Wind"

  uzon=u_int(:,:,0)
  uzon=dim_avg(u_int)

  print(uzon)

;===========================
; plot parameters
;===========================
 wks   = gsn_open_wks ("png", "conwomap" )         ; send graphics to PNG file

 res               = True            ; Plot mods desired
 res@gsnMaximize   = True            ; Maximize plot in frame
 res@cnFillOn      = True            ; Turn on contour fill
 res@tiMainString  = "Linear axis"   ; Main title

 plot = gsn_csm_contour(wks, uzon, res )

 res@tiMainString          = "Logarithmic axis"
 res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic

;---Set some axis labels, otherwise all we'll get is "10^3".
 res@tmYLMode              = "Explicit"
 res@tmYLValues            = (/300,500,700,1000/)
 res@tmYLLabels            = "" + res@tmYLValues

 plot = gsn_csm_contour(wks, uzon, res )
end