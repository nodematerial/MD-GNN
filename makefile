CONF_NAME :=

lmp:
	bash shell_script/lmp.sh ${CONF_NAME}


thermo:
	bash shell_script/thermo.sh ${CONF_NAME}

cna:
	bash shell_script/cna.sh ${CONF_NAME}

makegraph:
	bash shell_script/makegraph.sh ${CONF_NAME}

all:
	bash shell_script/lmp.sh ${CONF_NAME}
	bash shell_script/thermo.sh ${CONF_NAME}
	bash shell_script/cna.sh ${CONF_NAME}
	bash shell_script/makegraph.sh ${CONF_NAME}

remove:
	bash shell_script/remove.sh ${CONF_NAME}