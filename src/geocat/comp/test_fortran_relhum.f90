PROGRAM testRelhum

    IMPLICIT NONE
    DOUBLE PRECISION :: t, w, p, rh

    t = 18.D0+273.15D0
    w = 6/1000
    p = 1000.D0 * 100.D0

    CALL DRELHUM(t,w,p)


END PROGRAM testRelhum