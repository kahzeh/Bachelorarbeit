import subprocess 

x = '1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. f3 e5 7. Nb3 Be7 8.Be3 O-O 9. Qd2 Be6 10. O-O-O Nbd7 11. g4 b5 12. g5 Nh5 13. h4 Nb6 14. Kb1 Ng3 15. Rg1 Nxf1 16. Rdxf1 Nc4 17. Qe2 b4 18. Nd5 Bxd5 19. exd5 Nxe3 20. Qxe3 a5 21. Nd2 a4 22. f4 Ra5 23. f5 f6 24. gxf6 Bxf6 25. h5 Rxd5 26. h6 Rf7 27. Ne4 Kh8 28. hxg7+ Rxg7 29. Qh6 Rf7 30. Rg6 Bg7 31. Qh5 Qf8 32. Ng5 h6 33. Rxh6+ Bxh6 34. Nxf7+ Kh7 35. Qg6# 1-0'

s = subprocess.check_output(['pgn-extract', x ])

print(s)