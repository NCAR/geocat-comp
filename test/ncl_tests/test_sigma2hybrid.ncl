;***********************
; based on vert_2.ncl
; generates an output array and saves it
;***********************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;===========================

begin
;*************************************************
; read in data
;*************************************************
  in2 = addfile("u.89335.1_subset_time361.nc","r")
  u   = in2->u(:, 0:2, 0:1)
  ;u   = in2->u(:, 0, 0)

  in4 = addfile("ps.89335.1.nc","r")
  ps  = in4->ps(361, 0:2, 0:1)
  ;ps  = in4->ps(361, 0, 0) * 100   ; Pa
  sigma = in4->sigma

  ; Reorder
  u_reordered = u(xcoord|:, ycoord|:, sigma|:)      ;reorder
  ps_reordered = ps(xcoord|:, ycoord|:)  * 100      ;reorder   ; Pa

  p0     = 100000.   ; Pa

  hyam = (/ 0.0108093, 0.0130731, 0.03255911, 0.0639471 /)   ; hybrid coef  (sigma)
  hybm = (/ 0.0108093, 0.0173664, 0.06069280, 0.1158237 /)   ; hybrid coef  (pressure)

;***************************************************
; interpolate sigma to hybrid levels
;***************************************************

  print(u)
  print(u_reordered)
  print(ps_reordered)
  print(sigma)
  print(hyam)
  print(hybm)

  xh    = sigma2hybrid(u_reordered, sigma, hyam, hybm, p0, ps_reordered, 1)

  print(xh)

;===========================
; write output to netcdf
;===========================

   system("/bin/rm -f sigma2hybrid_output.nc")   ; remove any pre-existing file
   ncdf = addfile("sigma2hybrid_output.nc" ,"c")  ; open output netCDF file

;===================================================================
; create global attributes of the file (optional)
;===================================================================
   fAtt               = True            ; assign file attributes
   fAtt@title         = "sigma2hybrid output of u.89335.1.nc (vert_2 example)"
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

   ncdf->xh     = xh                          ; 3D

end
