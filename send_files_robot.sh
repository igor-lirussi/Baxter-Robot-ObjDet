# Read the remote IP address from ip_robot.txt
remote_ip=$(cat ip_robot.txt)

# Remote directory path
remote_dir="~/igor/Baxter-Robot-ObjDet/"

echo "Do you want to create also the subfolders in the remote?(You have to insert the password for each one!) (y/n)"
read user_input

if [ "$user_input" == "y" ]; then
	echo "Selected yes (y)."
	# Use mkdir to create missing directories on the remote server
	ssh "ruser@$remote_ip" "mkdir -p $remote_dir"
	for dir in $(find . -type d ! -wholename './.*'); do
		echo Creating: $remote_dir${dir#'./'}
		ssh "ruser@$remote_ip" "mkdir -p $remote_dir${dir#'./'}"
	done
fi

echo "Use scp to copy all files recursively to the remote server that should have the folders"
scp -r * "ruser@$remote_ip:$remote_dir"

echo "#### Done, press any key ####"
read -t 5 -n 1