---
layout: post
title: "Network Security and Threats"
---

Hello! Welcome to this post about network security and threats. This will cover different threats to networks and how these threats are defended from the network.

## Firewalls

A firewall is a security checkpoint which precents unauthorised access between 2 networks. Firewalls can be implemented both through hardware and through software.

A usual firewall contains 2 network cards (one for the internal network and one for the external network). Each data packet that attempts to pass between the two NICs is analysed against a set of rules, which decide whether the packet should be rejected or not.

## Packet filtering

Packet filtering allows packets to enter the network depending on their source IP and the port number. 

Packet filtering would only allow packets coming from IPs on the admins "permitted" list.

The internal network would also only allow certain protocols. Since each protocol uses a certain port number, the port number on the packet header would be checked and would be accordingly accepted, dropped or rejected. Dropped packets are quietly removed, while rejected packets cause a rejection notice to be sent back to the sender.

## Proxy servers

A proxy server intercepts all packets entering and leaving a network, which hides the true network addresses of the sender from the recipient. 

Proxy servers also hold a cache of commonly visited website, so it can return the web page without the need of having to reconnect to the Internet. This increases the speed of user access to web page data and reduced web traffic. If a web page is not in cache, then the proxy server makes the request on behalf of the user, using its own IP address.

Proxy servers are often used to filter requests providing administrative control over the content that users may demand. For example, a school may use a proxy server to monitor all requests and block undesirable websites. These proxies may also log user data with the requests.

![Alt Text](/assets/proxy-diagram.png) 

## Viruses and worm subclasses

Both viruses and worms can spread copies of themselves (self-replicate). A worm is a sub-class of a virus. While virus relies on other host files to be openeded to spread themselves, worms are standalone piece of software which can replicate themselves with no interaction from the user. 

Most viruses reside in memory when their host file is executed. Any uninfected file that is copied to memory when it runs becomes infected. Other viruses can reside in macro files, which are usually attached to word processing and spreadsheet data files.

Worms can reside in a data file of another application and will usually enter through a vulnerability in the machine or by tricking the user to open the file e.g through email attachments. Instead of infecting other files, worms can replicate themselves and send copies of itself to other users. 

Because of this, worms often use up a lot of bandwidth, system memory and network resources, causing the computers to slow and servers to stop responding.

## Trojans

A trojan hides itself as a useful program such as a game or a utility program you would want to have on your computer. When installed, the paylod gets released, often without notice. 

Trojans are commonly used to open a back door to your computer system, which can be exploited by the creator of the trojan. This can be used to harvest personal data or use your computer and network power to send thousands of spam emails to others. Groups of internet-enabled computers which are used like this are called bot-nets. Trojans, however, can not self-replicate themselves. 

## Protection against threats

Code quality is a primary vulnerability in a system.

Many malwares exploit a "buffer overflow", which happens when a value is written to a memory location too small to handle the value. This causes neighbouring locations, which aren't meant to have access to, to be overwritten. Overflow data is often interpreted as instructions, so viruses can take advantage of this by forcing the program to write something to memory.

To protect against social engineering, such as phishing attacks, spam filtering and education is the most effect method against this vulnerability.

Regular operating system and antivirus would also reduce the risk of attacks. Virus checkers usually scan for all types of viruses, but since new variants are created all the time, it is important that you update the virus checker.
