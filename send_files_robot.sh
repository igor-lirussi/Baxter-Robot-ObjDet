echo "Insert psw to access the robot account"
scp *.py ruser@$(cat ip_robot.txt):~/igor/Baxter-Robot-ObjDet
echo "#### Done, press any key ####"
read -t 5 -n 1