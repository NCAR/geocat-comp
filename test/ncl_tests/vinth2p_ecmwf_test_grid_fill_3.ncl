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
; Interpolate with no extrapolation
;-----------------------------------------------------------------

 Tp   = vinth2p (T,hyam,hybm,lev_p,PS,1,P0mb,1,False)
 Qp   = vinth2p (Q,hyam,hybm,lev_p,PS,1,P0mb,1,False)
 Zp   = vinth2p (Z,hyam,hybm,lev_p,PS,1,P0mb,1,False)

 Tp@long_name = "Temperature (K)"
 Qp@long_name = "Specific Humidity (g/kg)"
 Zp@long_name = "Geopotential (m)"

;-----------------------------------------------------------------
; Interpolate and extrapolation below surface pressure
;-----------------------------------------------------------------

 varflg = 1
 Tpx    = vinth2p_ecmwf (T,hyam,hybm,lev_p,PS,1,P0mb,1,True,varflg,TS,PHIS)
 varflg = 0
 Qpx    = vinth2p_ecmwf (Q,hyam,hybm,lev_p,PS,1,P0mb,1,True,varflg,TS,PHIS)
 varflg = -1
 Zpx    = vinth2p_ecmwf (Z,hyam,hybm,lev_p,PS,1,P0mb,1,True,varflg,TS,PHIS)
 print(Tpx)
;-----------------------------------------------------------------
; set the poisson_grid_fill input variables
;-----------------------------------------------------------------
  nscan = 2000             ; usually *much* fewer
  eps   = 0.001            ; variable depended
  gtype = True             ; Cyclic in longitude [global]
  guess = 0                ; use zonal means
  relc  = 0.6              ; standard relaxation coef
  opt   = 0

;-----------------------------------------------------------------
; Global grid: Fill in over land
;-----------------------------------------------------------------

  Tp_orig = Tp                        ; save for demo plot
  Qp_orig = Qp
  Zp_orig = Zp

  poisson_grid_fill( Tp, gtype, guess, nscan, eps, relc, opt)
  poisson_grid_fill( Qp, gtype, guess, nscan, eps, relc, opt)
  poisson_grid_fill( Zp, gtype, guess, nscan, eps, relc, opt)


;-----------------------------------------------------------------
; plot for compare
;-----------------------------------------------------------------

  LEV   = 950                              ; arbitrary level for demo

  wks   = gsn_open_wks("png","grid_fill")  ; send graphics to PNG file
  plot  = new(3,graphic)

  res                              = True
  res@gsnDraw                      = False
  res@gsnFrame                     = False
  res@cnInfoLabelOrthogonalPosF    = -0.07

  res@cnInfoLabelOn       = False           ; turn off cn info label
  res@cnFillOn            = True            ; turn on color
  res@cnFillPalette       = "amwg"          ; set color map

  res@lbLabelBarOn        = False           ; turn off individual cb's
  res@mpCenterLonF        = 180.
  res@mpFillOn            = False

  resP    = True
 ;resP@gsnPanelMainString = title               ; uncomment to add title
  resP@gsnMaximize        = True                ; make large
  resP@gsnPanelLabelBar   = True                ; add common colorbar
 ;resP@lbLabelStride      = 2                   ; force every other label
 ;resP@lbLabelFontHeightF = 0.0125              ; make labels smaller [0.2 default]

; Temperature

  res@gsnRightString = "Original: "+LEV+"hPa"
  plot(0) = gsn_csm_contour_map(wks,Tp_orig(0,{LEV},:,:),res)

  res@gsnRightString = "vinth2p with extrapolation"
  plot(1) = gsn_csm_contour_map(wks,Tpx(0,{LEV},:,:),res)

  res@gsnRightString = "poisson_grid_fill"
  plot(2) = gsn_csm_contour_map(wks,Tp(0,{LEV},:,:),res)
  gsn_panel(wks,plot,(/3,1/),resP)

; Specific humidity

  res@gsnRightString = "Original: "+LEV+"hPa"
  plot(0) = gsn_csm_contour_map(wks,Qp_orig(0,{LEV},:,:),res)

  res@gsnRightString = "vinth2p with extrapolation"
  plot(1) = gsn_csm_contour_map(wks,Qpx(0,{LEV},:,:),res)

  res@gsnRightString = "poisson_grid_fill"
  plot(2) = gsn_csm_contour_map(wks,Qp(0,{LEV},:,:),res)
  gsn_panel(wks,plot,(/3,1/),resP)

; Geopotential

  res@gsnRightString = "Original: "+LEV+"hPa"
  plot(0) = gsn_csm_contour_map(wks,Zp_orig(0,{LEV},:,:),res)

  res@gsnRightString = "vinth2p with extrapolation"
  plot(1) = gsn_csm_contour_map(wks,Zpx(0,{LEV},:,:),res)

  res@gsnRightString = "poisson_grid_fill"
  plot(2) = gsn_csm_contour_map(wks,Zp(0,{LEV},:,:),res)
  gsn_panel(wks,plot,(/3,1/),resP)


end