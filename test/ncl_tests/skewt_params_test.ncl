;**************************************************
; skewt_2.ncl
;
; Concepts illustrated:
;   - Drawing Skew-T plots
;   - Drawing a Skew-T plot with and without wind barbs
;   - Customizing the background of a Skew-T plot
;*************************************************
;
; This file is loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
;
; This file still has to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/skewt_func.ncl"
;***********************************************
 begin

; --- Read Data; no winds; set to _FillValue------------;
  diri = "./"
  fili = "sounding.testdata"
  nlvl = 30
  ncol = 16
  TestData = asciiread (diri+fili , (/nlvl,ncol/), "float")

  p    = TestData (:,1)
  z    = TestData (:,2)
  q    = TestData (:,4)
  tc   = TestData (:,5) + 2.    ; for demo purposes
  tdc  = TestData (:,9)
                                ; create BOGUS winds
  wspd = new ( nlvl, "float", -999.)  ; wspd@_FillValue = -999.
  wdir = new ( nlvl, "float", -999.)  ; wdir@_FillValue = -999.

;*************************
; create plot
;*************************
  wks  = gsn_open_wks ("png", "skewt")  ; send graphics to PNG file

; --- Create background skew-T; plot sounding ---------------------

  skewtOpts                 = True
  skewtOpts@DrawColAreaFill = True    ; default is False
  skewtOpts@tiMainString    = "Raob Data; No Winds"

  dataOpts   = False        ; no options

;  dataOpts@PrintZ = False
  skewt_bkgd = skewT_BackGround (wks, skewtOpts)
  skewt_data = skewT_PlotData   (wks, skewt_bkgd, p,tc,tdc,z \
                                    , wspd,wdir, dataOpts)
  draw (skewt_bkgd)
  draw (skewt_data)
  frame(wks)

;


  ;===========================
  ; write output to netcdf
  ;===========================

     system("/bin/rm -f skewt_params_output.nc")   ; remove any pre-existing file
     ncdf = addfile("skewt_params_output.nc" ,"c")  ; open output netCDF file

  ;===================================================================
  ; create global attributes of the file (optional)
  ;===================================================================
     fAtt               = True            ; assign file attributes
     fAtt@title         = "SkewT output of sounding.testdata (skewt_2 examples)"
     fAtt@source_file   =  "original-file.nc"
     fAtt@Conventions   = "None"
     fAtt@creation_date = systemfunc ("date")
     fileattdef( ncdf, fAtt )            ; copy file attributes

  ;===================================================================
  ; output variables directly; NCL will call appropriate functions
  ; to write the meta data associated with each variable
  ;===================================================================
    ncdf->Plcl = skewt_data@Plcl
    ncdf->Pwat = skewt_data@Pwat
    ncdf->Tlcl = skewt_data@Tlcl
    ncdf->Shox = skewt_data@Shox
    ncdf->Cape = skewt_data@Cape

end
